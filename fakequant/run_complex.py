import time
import torch
import torch.nn as nn
import torch.nn as nn
import sys
sys.path.append('../methods')
from methods.complex_utils.complex_v1 import complex_quant
#from complex_utils.complex_v2 import complex_quant
from modelutils import find_layers
DEBUG=False

def get_model(model):
    import torch
    print("Loading model ...")
    print(model)
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" or "Llama" in model:

        from transformers import LlamaForCausalLM
        from transformers import AutoModelForCausalLM
        #model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    return model


'''
The function is employed to calibrate and quantize models layer by layer.
'''
@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" or "Llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb=model.model.rotary_emb.to(dev)
    else :
        raise ValueError("Model not supported")
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}
    if "llama" in args.model.lower():
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_embeddings"]=kwargs["position_embeddings"]
                raise ValueError
    else:
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                #cache["position_embeddings"]=kwargs["position_embeddings"]
                raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" or "Llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        #model.model.rotary_emb=model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    if "llama" in args.model.lower():
        position_embeddings=cache["position_embeddings"]
    print("Ready.")
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            
            gptq[name] = complex_quant(
                subset[name],
                cluster_m=args.cluster_m,
                cluster_p=args.cluster_p,
                group_size=args.groupsize,
                #disable_complex=True
                disable_complex=args.disable_complex,
            )

            if DEBUG:
                torch.save(subset[name].bias, f"./output/{(args.model.split('/')[-1])}_layers_{i}_{name}_bias.pt")
        def add_batch(name,temp):
            def tmp(module, inp, out):
                if DEBUG:
                    temp_inp=inp[0]
                    if len(temp_inp.shape)!=3:
                        temp_inp=temp_inp.unsqueeze(0)
                    temp.append(temp_inp)
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        inps_dict={}
        for name in gptq:
            inps_dict[name]=[]
            handles.append(subset[name].register_forward_hook(add_batch(name,inps_dict[name])))

        for j in range(args.nsamples):
            #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            if "llama" in args.model.lower():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,position_embeddings=position_embeddings)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        if DEBUG:
            for name in gptq:
                torch.save(torch.stack(inps_dict[name],dim=0).squeeze(),f"./output/{args.model.split("/")[-1]}_layers_{i}_{name}_inp.pt")
        for h in handles:
            h.remove()

        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            if DEBUG:
                #torch.save(gptq[name].H, f"./output/{args.model.split("/")[-1]}_layers_{i}_{name}_H.pt")
                torch.save(gptq[name].layer.weight.data, f"./output/{args.model.split("/")[-1]}_layers_{i}_{name}_weight.pt")
                pass
            info = gptq[name].fasterquant(
                blocksize=args.blocksize,
                percdamp=args.percdamp, 
            )

            gptq[name].free()
            if DEBUG:
                pass
                #torch.save(gptq[name].layer.weight.data, f"./output/{args.model.split("/")[-1]}_layers_{i}_{name}_quant_weight.pt")
        for j in range(args.nsamples):
            if "llama" in args.model.lower():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,position_embeddings=position_embeddings)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            #outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4","pileval"],
        help="Where to extract calibration data from.",
    )

    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )

    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="groupsize for cluster.",
    )

    parser.add_argument(
        "--cluster_m",
        type=int,
        default=2,
        help="cluster center for amplitude.",
    )

    parser.add_argument(
        "--cluster_p",
        type=int,
        default=8,
        help="cluster center for phase.",
    )
    parser.add_argument(
        "--disable_complex",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    # parser.add_argument(
    #     "--rank",
    #     type=int,
    #     default=2,
    #     help="Rank of the low-rank approximation.",
    # )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
 
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()
    groupsize = args.blocksize

    device = args.device
    save_title = f"{args.model}_{args.dataset}__percdamp{args.percdamp}_blocksize{args.blocksize}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    else: # braq
        model = get_model(args.model)
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quant_sequential(model, dataloader, device)
        print("quantization time:", time.time() - tick, "s")

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" or "Llama" in args.model:
            from eval_ppl_utils import llama_eval
            llama_eval(model, testloader, device, dataset, args.log_wandb)
        #break
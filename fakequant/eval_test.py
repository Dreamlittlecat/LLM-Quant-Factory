import time
import torch
import torch.nn as nn
import os
import sys
sys.path.append('./eval_my')

#from methods.complex_utils.complex_v1 import complex_quant
#from methods.complex_utils.complex_v2 import complex_quant
# from methods.complex_utils.complex_v3 import complex_quant
from modelutils import find_layers


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
        #较早版本的transformers
        # from transformers import OPTForCausalLM
        # model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        os.environ["HF_SKIP_CHECK_PICKLE"] = "1"
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model.lower():

        #from transformers import LlamaForCausalLM
        from transformers import AutoModelForCausalLM
        #model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    
    else:
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", local_files_only=True)
        #print("model max len:",model.config.max_position_embeddings,model.seqlen)
        model.seqlen = 2048
        #raise ValueError("Model not supported")
    print("Model loaded.")
    print("model", model)
    return model

if __name__ == "__main__":
    import argparse

# ...existing code...
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    args = parser.parse_args()
    from datautils import *


    device =  "cuda:0"
    model = get_model(args.model)
    model.eval()
    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
    )
        print(dataset)
        if "opt" in args.model.lower():
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model.lower():
            from eval_ppl_utils import llama_eval
            llama_eval(model, testloader, device, dataset, args.log_wandb)
        elif "qwen" in args.model.lower():
            from eval_ppl_utils import qwen_eval
            #raise NotImplementedError
            qwen_eval(model, testloader, device, dataset, args.log_wandb)
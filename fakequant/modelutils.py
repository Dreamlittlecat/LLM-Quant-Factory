import torch
import torch.nn as nn
DEV = torch.device('cuda:0')
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
    
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
    elif "llama" in model.lower():

        #from transformers import LlamaForCausalLM
        from transformers import AutoModelForCausalLM
        #model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    
    else:
        from transformers import AutoModelForCausalLM
        model=AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        #print("model max len:",model.config.max_position_embeddings,model.seqlen)
        model.seqlen = 2048
        #raise ValueError("Model not supported")
    print("Model loaded.")
    print("model", model)
    return model
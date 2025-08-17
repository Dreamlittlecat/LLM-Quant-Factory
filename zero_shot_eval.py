import os
import torch
torch.backends.cudnn.benchmark = True
from lm_eval import evaluator
from pprint import pprint,pformat
from models.LMClass import LMClass2,LMClass
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def evaluate(model, tasks, num_fewshot=0, limit=None):
    """
    Evaluate the model on the specified tasks.
    :param model: The model to evaluate.
    :param tasks: A list of task names to evaluate on.
    :param num_fewshot: The number of few-shot examples to use for evaluation.
    :param limit: The maximum number of examples to evaluate on each task.
    :return: A dictionary containing the evaluation results.
    """
    results = {}
    t_results = evaluator.simple_evaluate(
        model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    results.update(t_results)
    return results

def get_args():
    """
    Parse command-line arguments.
    :return: The parsed arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a model on specific tasks.")
    parser.add_argument(
        "--model",
        type=str,
        default="./",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./",
        help="save log .",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="openbookqa",#"boolq,winogrande,arc_easy,arc_challenge",#--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
        help="Comma-separated list of tasks to evaluate on.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples to use for evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate on each task.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for evaluation (e.g., 'cuda:0' or 'cpu').",
        
    )
    return parser.parse_args()
def main():
    args=get_args()
    model=AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device,torch_dtype=torch.float16)
    tokenizer=AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
    # model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cpu',torch_dtype=torch.float16)
    model=LMClass2(model,tokenizer)
    tasks=args.tasks
    results = evaluate(model, tasks, num_fewshot=args.num_fewshot, limit=args.limit)

    model_name = args.model.split("/")[-1]
    # save_dir = os.path.dirname(args.model)
    save_dir = args.log_dir+"/"+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{model_name}_results.log")
    with open(save_path, "w") as f:
        f.write(pformat(results))
    print(f"Results saved to {save_path}")
    
if __name__ == "__main__":
    main()
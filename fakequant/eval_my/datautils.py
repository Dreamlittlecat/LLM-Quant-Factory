import os
import numpy as np
import torch
import json
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
import random
from itertools import chain
from transformers import DataCollatorForLanguageModeling, default_data_collator
import transformers
import tqdm

num_proc=64
use_cache = False

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_qat_dataset(name, tokenizer, model="llama3", seed="random"):
    cache_file=f'cache/{name}_{model}_{seed}.pt'
    if use_cache:
        try:
            return torch.load(cache_file), default_data_collator
        except:
            pass
    if name == "wikitext2":
        data, data_collator = get_wikitext2_train(tokenizer=tokenizer)
    elif name == "c4":
        data, data_collator = get_c4_train(tokenizer=tokenizer)
    elif name == "c4_wiki":
        data, data_collator = get_c4_wiki_train(tokenizer=tokenizer)
    elif name == "tulu":
        data, data_collator = get_tulu_train(tokenizer=tokenizer)
    elif name == "alpaca_clean":
        data, data_collator = get_alpaca_clean_train(tokenizer=tokenizer)
    elif name == "wikitext2_test":
        data, data_collator = get_wikitext2_test4train(tokenizer=tokenizer)

    if use_cache:
        directory='/'.join(cache_file.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(data, cache_file)

    return data, data_collator

def get_eval_loaders(name, tokenizer):
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
                print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            except AttributeError:
                pass
                print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(tokenizer)
        return get_ptb(tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(tokenizer)
        return get_c4(tokenizer)


def get_wikitext2_train(tokenizer, seed=0, seqlen=2048):
    # dataset = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="train",
    # )
    dataset = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train')

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(dataset["text"])
                ],
            },
        )

    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator

def get_wikitext2_test4train(tokenizer, seed=0, seqlen=2048):
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    # dataset = load_from_disk('ppl_datasets/wikitext/wikitext-2-raw-v1/train')
    # dataset = load_dataset('/root/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1', split='test')

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(dataset["text"])
                ],
            },
        )

    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator

def get_alpaca_clean_train(tokenizer, seed=0, seqlen=2048):

    dataset = load_dataset('/home/hhj/workspace/Topic/W1W8LLM/datasets/yahma/alpaca-cleaned', split="train")

    wikitext_dataset = datasets.Dataset.from_dict(
            {
                "output": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(dataset["output"])
                ],
            },
        )

    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["output"])
        .add_column(
            name="url",
            column=wikitext_dataset["output"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "output" if "output" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator

def get_tulu_train(tokenizer, seed=0, seqlen=2048):
    # dataset = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="train",
    # )
    
    def filter_tulu_flan_v2(example):
        text = ''
        for i in example['messages']:
            text += i["content"]
            text += "\n\n"
        example['text'] = text
        return example
    # def filter_tulu_flan_v2(example):
    #     text = []
    #     for i in example['messages']:
    #         text.append(i["content"])
    #     example['text'] = text
    #     return example


    dataset = load_from_disk("/home/hhj/workspace/Topic/LLaMA3-Quant/IR-QLoRA/tulu-v2-sft-mixture")['train']
    # wiki = load_from_disk('ppl_datasets/wikitext/wikitext-2-raw-v1/train')
    # import pdb; pdb.set_trace()
    dataset = dataset.map(filter_tulu_flan_v2, remove_columns=['id','dataset','messages'], num_proc=32)
    # wikitext_dataset = datasets.Dataset.from_dict({"text": ["\n\n".join(dataset_["messages"])],},)
    # wikitext_dataset = datasets.Dataset.from_dict({"text": ["\n\n".join(wiki["text"])],},)

    # wikitext_dataset = datasets.Dataset.from_dict(
    #         {
    #             "text": [
    #                 # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
    #                 "\n\n".join(dataset["text"])
    #             ],
    #         },
    #     )
    wikitext_dataset = dataset
    # Hacks to get around the `remove_columns` to be used later.
    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )
    column_names = list(wikitext_dataset.features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # import pdb; pdb.set_trace()
    tokenized_datasets = wikitext_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset, default_data_collator


def get_alpaca_clean_train_(tokenizer, seed=0, seqlen=2048):

    dataset = load_dataset("/home/hhj/workspace/Topic/W1W8LLM/datasets/yahma/alpaca-cleaned")
    dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])

    ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    }
    def extract_alpaca_dataset(example):
        if example.get("input", "") != "":
            prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
        else:
            prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
        return {'input': prompt_format.format(**example)}
    # Remove unused columns.
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
    )
    
    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    import copy
    from torch.nn.utils.rnn import pad_sequence
    
    class DataCollatorForCausalLM(object):
        tokenizer: transformers.PreTrainedTokenizer
        source_max_len: int
        target_max_len: int
        train_on_source: bool
        predict_with_generate: bool

        def __call__(self, instances):
            # Extract elements
            sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
            targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
            # Tokenize
            tokenized_sources_with_prompt = self.tokenizer(
                sources,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = self.tokenizer(
                targets,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            # Build the input and labels for causal LM
            input_ids = []
            labels = []
            for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
            ):
                if not self.predict_with_generate:
                    input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                    if not self.train_on_source:
                        labels.append(
                            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                        )
                    else:
                        labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
                else:
                    input_ids.append(torch.tensor(tokenized_source))
            # Apply padding
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
            data_dict = {
                'input_ids': input_ids,
                'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            }
            if labels is not None:
                data_dict['labels'] = labels
            return data_dict

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=384,
        target_max_len=128,
        train_on_source=False,
        predict_with_generate=False,
    )
    return processed_dataset, data_collator

def get_c4_train(tokenizer, seed=0, seqlen=2048):
    # raw_datasets = load_dataset(
    #     "allenai/c4",
    #     #"allenai--c4",
    #     data_files={
    #         "train": "en/c4-train.00000-of-01024.json.gz",
    #         "validation": "en/c4-validation.00000-of-00008.json.gz",
    #     },
    # )
    # raw_datasets = load_dataset('/home/hhj/.cache/huggingface/datasets/allenai___c4')
    raw_datasets = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4')
    # _wikitext_dataset = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="test",
    #     )
    _wikitext_dataset = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/test')
    # Hacks to be consistent with other works' preprocessing.
    wikitext_dataset = datasets.Dataset.from_dict(
        {
            "text": [
                # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                "\n\n".join(_wikitext_dataset["text"])
            ],
        },
    )

    wikitext_dataset = (
        wikitext_dataset  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset["text"])
        .add_column(
            name="url",
            column=wikitext_dataset["text"])
    )

    raw_datasets["wikitext"] = wikitext_dataset

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset["train"], default_data_collator




def filter_c4_train(example, tokenizer, source_min_len, source_max_len):
    messages = example['text']
    input_text = ""
    for message in messages:
        input_text += message
    input_token = tokenizer(input_text)
    return len(input_token['input_ids']) > source_min_len and len(input_token['input_ids']) < source_max_len

def get_c4_wiki_train(tokenizer, seed=0, seqlen=2048):
    # raw_datasets = load_dataset(
    #     "allenai/c4",
    #     #"allenai--c4",
    #     data_files={
    #         "train": "en/c4-train.00000-of-01024.json.gz",
    #         "validation": "en/c4-validation.00000-of-00008.json.gz",
    #     },
    # )
    # raw_datasets = load_dataset('/home/hhj/.cache/huggingface/datasets/allenai___c4')
    # raw_datasets.save_to_disk('/home/hhj/workspace/Topic/W1W8LLM/datasets/c4')
    raw_datasets = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4')
    # _wikitext_dataset_train = load_dataset(
    #     "wikitext",
    #     "wikitext-2-raw-v1",
    #     split="train",
    #     )
    wikitext_dataset_eval = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        )
    wikitext_dataset_train = load_from_disk('../eval_my/ppl_datasets/wikitext/wikitext-2-raw-v1/train')
    #wikitext_dataset_eval = load_dataset('/root/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1', split='test')
    # Hacks to be consistent with other works' preprocessing.
    # wikitext_dataset_train = datasets.Dataset.from_dict(
    #     {
    #         "text": [
    #             # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
    #             "\n\n".join(_wikitext_dataset_train["text"])
    #         ],
    #     },
    # )
    # wikitext_dataset_eval = datasets.Dataset.from_dict(
    #     {
    #         "text": [
    #             # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
    #             "\n\n".join(_wikitext_dataset_eval["text"])
    #         ],
    #     },
    # )

    # wikitext_dataset_train = (
    #     wikitext_dataset_eval  # type: ignore
    #     .add_column(
    #         name="timestamp",
    #         column=[None for _ in range(len(wikitext_dataset_eval["text"]))])
    #     .add_column(
    #         name="url",
    #         column=wikitext_dataset_eval["text"])
    # )
    wikitext_dataset_train = (
        wikitext_dataset_train  # type: ignore
        .add_column(
            name="timestamp",
            column=[None for _ in range(len(wikitext_dataset_train["text"]))])
        .add_column(
            name="url",
            column=wikitext_dataset_train["text"])
    )
    wikitext_dataset_eval = (
        wikitext_dataset_eval  # type: ignore
        .add_column(
            name="timestamp",
            column=wikitext_dataset_eval["text"])
        .add_column(
            name="url",
            column=wikitext_dataset_eval["text"])
    )
    
    # raw_datasets["train"] = concatenate_datasets([
    #     raw_datasets["train"].filter(filter_c4_train, fn_kwargs=({'tokenizer': tokenizer, 'source_min_len': 384, 'source_max_len': float("inf")}), num_proc=num_proc),
    #     wikitext_dataset_train,
    #     wikitext_dataset_train])

    # raw_datasets["train"] = concatenate_datasets([
    #     raw_datasets["train"].select(random.sample(list(range(len(raw_datasets["train"]))), 200000)),
    #     wikitext_dataset_train,
    #     wikitext_dataset_train,
    #     wikitext_dataset_train,
    #     wikitext_dataset_train])
    raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], wikitext_dataset_train])
    # raw_datasets["train"] = concatenate_datasets([
    #     raw_datasets["train"].select(random.sample(list(range(len(raw_datasets["train"]))), 75000)),
    #     raw_datasets["validation"],
    #     wikitext_dataset_train,
    #     wikitext_dataset_eval])
    raw_datasets["wikitext"] = wikitext_dataset_eval
    raw_datasets["train"][0]  # fast
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)
    raw_datasets["train"][0]  # up to 10x slower
    raw_datasets["train"] = raw_datasets["train"].flatten_indices()  # rewrite the shuffled dataset on disk as contiguous chunks of data
    raw_datasets["train"][0]  # fast again

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=num_proc
    )

    block_size = 2048

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )
    return processed_dataset["train"], default_data_collator


def get_wikitext2(tokenizer, seqlen=2048):
    testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    # testdata = load_dataset('/root/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1', split='test')
    # testdata = load_dataset('/root/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1', split='validation')

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return testenc

def get_ptb(tokenizer, seqlen=2048):
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    return testenc

save_seed = False

def get_c4(tokenizer, seqlen=2048):
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    valdata = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4/validation')
    # val_data = valdata.filter(filter_c4_train, fn_kwargs=({'tokenizer': tokenizer, 'source_min_len': 2048, 'source_max_len': float("inf")}), num_proc=64)
    # val_data = load_from_disk('/home/hhj/workspace/Topic/W1W8LLM/datasets/my_c4/val_2048')
    import random
    random.seed(0)
    # index_list = random.sample(list(range(len(val_data))), 256)
    valenc = []
    if save_seed:
        index_list = []
        tmp_list = []
    else:
        index_list = torch.load('../eval_my/c4_seed_index.pth')
        tmp_list = torch.load('../eval_my/c4_seed_tmp.pth')
    for _ in range(256):
        if save_seed:
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    tmp_list.append(i)
                    break
            # tmp = tokenizer(val_data[index_list[_]]['text'], return_tensors='pt')
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            index_list.append(i)
        else:
            i = index_list[_]
            tmp = tmp_list[_]
        j = i + seqlen
        if save_seed:
            valenc.append(tmp.input_ids[:, i:j])
        else:
            valenc.append(tokenizer(valdata[tmp]['text'], return_tensors='pt').input_ids[:, i:j])
    if save_seed:
        torch.save(index_list, '../eval_my/c4_seed_index.pth')
        torch.save(tmp_list, '/eval_my/c4_seed_tmp.pth')
        breakpoint()
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc 

def get_ptb_new(tokenizer, seqlen=2048):
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    return testenc

def get_c4_new(tokenizer, seqlen=2048):
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    valdata = load_from_disk('../eval_my/ppl_datasets/allenai/c4/allenai--c4/validation')

    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return valenc


if __name__ == "__main__":
    
    # get_tulu_train(None)
    get_c4_wiki_train(None)




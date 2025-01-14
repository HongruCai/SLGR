import os
from torch.utils.data import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding, GenerationConfig
import torch
import random
# import deepspeed
from typing import Dict, List
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
import argparse
import ujson
from tqdm import tqdm

class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, docid_to_smtid, query_to_docid, max_length, fraction=None):
        self.tokenizer = tokenizer
        self.max_source_len = max_length
        self.fraction = fraction
        with open(docid_to_smtid) as fin:
            docid2smtid = ujson.load(fin)

        with open(query_to_docid, "r") as f:
            query_to_docid = ujson.load(f)
        
        self.examples = []
        for query, docids in query_to_docid.items():
            for docid in docids:
                smtid = docid2smtid[docid]
                self.examples.append((query, smtid))
        if self.fraction is not None:
            subset_size = int(len(self.examples) * self.fraction)
            sampled_indices = random.sample(range(len(self.examples)), subset_size)
            self.examples = [self.examples[i] for i in sampled_indices]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        source_text, ids = self.examples[idx]
        source_text = source_text.strip()
        return preprocess_function(source_text, ids, self.tokenizer, self.max_source_len)


def preprocess_function(prefix_text, ids, tokenizer, max_length):
    target_text = " ".join(f"c_{c}" for c in ids[1:])
    response_text = f"{prefix_text} {target_text}</s>"

    input = tokenizer(
        response_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    input_ids = input["input_ids"]
    labels = input_ids.copy()

    prefix_tex = tokenizer(
        prefix_text,
        return_tensors=None,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    output_start_index = len(prefix_tex)

    labels[:output_start_index] = [-100] * output_start_index

    return {
        "input_ids": input_ids,
        "attention_mask": input["attention_mask"],
        "labels": labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llama")

    parser.add_argument('--docid_to_smtid', type=str, default='../data/MSMARCO/filtered_docid_to_smtid.json', help='docid to smtid mapping')
    parser.add_argument('--query_to_docid', type=str, default='../data/MSMARCO/filtered_query_to_docid.json', help='query to docid mapping')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--train_epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_steps', type=int, default=1000000, help='max steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--wandb_log_freq', type=int, default=5, help='wandb log frequency')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='no', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=500, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--fraction', type=float, default=None, help='fraction of dataset to use')
    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()

    print('training on: ', train_args.docid_to_smtid)

    model_name = train_args.model_name

    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    source_length = train_args.source_length

    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] + '_frac' + str(train_args.fraction)
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if ddp:
        device_map = {"": local_rank}
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    config = LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, 
                                             torch_dtype=torch_dtype,
                                             config=config,
                                             device_map=device_map)
    extra_tokens = []
    for count in range(256):
        extra_tokens.append('c_'+str(count))
    print('number of extra tokens: ', len(extra_tokens))
    tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    reporter = "none"

    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_args.train_epoch,
        # max_steps=1000,      
        per_device_train_batch_size=train_batch_size, 
        per_device_eval_batch_size=train_batch_size, 
        dataloader_num_workers=10,

        #optim = "adamw_8bit",   
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        # weight_decay=0.01,   
                   
        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        save_strategy=train_args.save_strategy,
        save_total_limit=train_args.save_total_limit,

        logging_steps=train_args.logging_steps,

        # deepspeed=train_args.deepseed_config,
        # gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        fp16=train_args.float16,
        bf16=train_args.bf16,

        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        save_only_model=True,
    )

    train_dataset = LLaMaDataset(tokenizer, train_args.docid_to_smtid, train_args.query_to_docid, source_length, fraction=train_args.fraction)

    print('training dataset size: ', len(train_dataset))
    print(train_dataset[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir_name)


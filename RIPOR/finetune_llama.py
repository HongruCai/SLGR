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

class LLaMaDataset(Dataset):
    def __init__(self, tokenizer, source_file, target_file, max_length, subset_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset_size = subset_size

        with open(source_file, 'r', encoding="utf-8") as f:
            source_data = f.readlines()

        with open(target_file, 'r', encoding="utf-8") as f:
            target_data = f.readlines()
        assert len(source_data) == len(target_data), "Source and target files must have the same number of lines."

        self.dataset = list(zip(source_data, target_data))
        if self.subset_size is not None:
            indices = list(range(len(self.dataset)))
            sampled_indices = random.sample(indices, self.subset_size)
            self.dataset = [self.dataset[i] for i in sampled_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source_text, target_text = self.dataset[idx]
        source_text = source_text.strip()
        target_text = target_text.strip()
        return preprocess_function(source_text, target_text, self.tokenizer, self.max_length)


def preprocess_function(prefix_text, target_text, tokenizer, max_length):
    response_text = f"{prefix_text}{target_text}</s>"

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

    parser.add_argument('--data_path', type=str, default='data', help='param data path')
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
    parser.add_argument('--save_strategy', type=str, default='epoch', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=5, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=100, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    
    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.data_path

    print('training on: ', data_path)

    model_name = train_args.model_name

    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    source_length = train_args.source_length

    output_dir_name = train_args.output_dir + '/' + train_args.model_name.split('/')[-1] 
    
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
                                            #load_in_8bit=True,
                                            #load_in_4bit=True,
                                            #bnb_4bit_compute_dtype=torch_dtype,
                                             device_map=device_map)

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
        # max_steps=train_args.max_steps,      
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
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        fp16=train_args.float16,
        bf16=train_args.bf16,

        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        save_only_model=True,
    )

    train_dataset = LLaMaDataset(tokenizer, data_path + '/llama_train.source', data_path + '/llama_train.target', source_length)

    print('training dataset size: ', len(train_dataset))

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


import os
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
import torch
import logging
import argparse
from utils import QueryEvalCallback, TrainerwithTemperature, T5Dataset, T5OriDataset
from peft import TaskType, LoraConfig, get_peft_model, PeftModel
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model")

    parser.add_argument('--docid_to_smtid', type=str, default='../data/MSMARCO/filtered_docid_to_smtid.json', help='docid to smtid mapping')
    parser.add_argument('--query_to_docid', type=str, default='../data/MSMARCO/filtered_query_to_docid.json', help='query to docid mapping')
    parser.add_argument('--output_dir', type=str, default='output/', help='output directory')
    parser.add_argument('--model_name', type=str, default='t5-base', help='model name')
    parser.add_argument('--train_epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--source_length', type=int, default=128, help='source length')
    parser.add_argument('--target_length', type=int, default=36, help='target length')
    parser.add_argument('--gen_len', type=int, default=20, help='generation length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--eval_strategy', type=str, default='epoch', help='evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='no', help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=1, help='save total limit')
    parser.add_argument('--logging_steps', type=int, default=1000, help='logging steps')
    parser.add_argument('--deepseed_config', type=str, default=None, help='deepspeed config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--float16', action='store_true', help='use float16')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--lora', action='store_true', help='use lora')

    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    data_path = train_args.docid_to_smtid

    print('training on: ', data_path)

    model_name = train_args.model_name

    train_epoch = train_args.train_epoch
    learning_rate = train_args.learning_rate
    train_batch_size = train_args.train_batch_size
    source_length = train_args.source_length
    target_length = train_args.target_length
    gen_len = train_args.gen_len


    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    output_dir_name = train_args.output_dir + '/' + train_args.model_name

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    if train_args.float16:
        torch_dtype = torch.float16
    elif train_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torch_dtype=torch_dtype,
                                                       config=config)

    extra_tokens = []
    for count in range(256):
        extra_tokens.append('c_'+str(count))
    print('number of extra tokens: ', len(extra_tokens))
    tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if train_args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q","v"],
            modules_to_save=["embed_tokens"],
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    reporter = "none"

    training_args = TrainingArguments(
        output_dir=output_dir_name,

        num_train_epochs=train_epoch,
        # max_steps=10000,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        dataloader_num_workers=10,

        # optim='adafactor',
        warmup_ratio=train_args.warmup_ratio,
        learning_rate=learning_rate,
        # weight_decay=0.01,

        logging_dir=output_dir_name+'/logs/',
        report_to=reporter,
        # eval_steps=1000,

        save_strategy=train_args.save_strategy,
        # save_steps=1000,
        save_total_limit=train_args.save_total_limit,
        # load_best_model_at_end=True,
        logging_steps=train_args.logging_steps,
        deepspeed=train_args.deepseed_config,

        save_only_model=True,

        fp16=train_args.float16,
        bf16=train_args.bf16,
    )
    model.config.use_cache = False

    # train_dataset = T5OriDataset(tokenizer, docid_to_smtid=train_args.docid_to_smtid, query_to_docid=train_args.query_to_docid ,max_source_len=source_length, max_target_len=target_length)
    train_dataset = T5Dataset(tokenizer, docid_to_smtid=train_args.docid_to_smtid, query_to_docid=train_args.query_to_docid ,max_source_len=source_length, max_target_len=target_length)

    print(train_dataset[0])
    print(len(train_dataset))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=source_length)

    os.makedirs(output_dir_name, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir_name)
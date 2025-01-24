import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
import os
from peft import LoraConfig, get_peft_model, TaskType
 
def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 with multiple GPUs and Weights & Biases support")
    

    # parser.add_argument("--train_file", type=str, required=True, help="Path to the training file (in CSV/JSON format)")
    
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model and results")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    #parser.add_argument("--project_name", type=str, default="Scaling", help="Weights & Biases project name")
    # parser.add_argument("--scenario_name", type=str, default="T5_scaling", help="Weights & Biases project name")
    # parser.add_argument("--experiment_name", type=str, default="T5_small", help="Weights & Biases project name")
    # parser.add_argument("--run_dir", type=str, default='/T5_scaling_wandb' , help="Weights & Biases project name")

    # parser.add_argument("--fp16", action="store_true", help="Use mixed precision (fp16)")
    #parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model_size", type=str, default='google-t5/t5-small', help="T5 model size: t5-small, t5-base, t5-large, etc.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--local-rank', type=int, default=0, help='local rank')
    args = parser.parse_args()
    return args

def preprocess_function(examples, tokenizer):
    inputs = examples['source']  
    targets = examples['target'] 
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def load_data_stream(source_path, target_path, num_samples=None):
    source_file = open(source_path, "r", encoding="utf-8")
    target_file = open(target_path, "r", encoding="utf-8")

    for idx, (src_line, tgt_line) in enumerate(zip(source_file, target_file)):
        if num_samples and idx >= num_samples:
            break
        src_line = src_line.strip()
        tgt_line = tgt_line.strip()
        if tgt_line.endswith(' @@'):
            tgt_line = tgt_line[:-3] + ' ' + '<extra_id_99>'
        yield {"source": src_line, "target": tgt_line}

    source_file.close()
    target_file.close()

def main():
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    output_dir = os.path.join(
        args.output_dir, f"{args.model_size.split('/')[-1]}")
    os.makedirs(output_dir, exist_ok=True)
    print('output_dir', output_dir)

    # if local_rank == 0:
    #         wandb.login()
    #         wandb.init(project='SL', name=output_dir)

    model = T5ForConditionalGeneration.from_pretrained(args.model_size)
    tokenizer = T5Tokenizer.from_pretrained(args.model_size)

    
    source_path = "../data/training/t5_train.source"
    target_path = "../data/training/t5_train.target"

    # Use the custom loading function with load_dataset for streaming
    # dataset = load_dataset("json", data_files={"train": [source_path, target_path]}, streaming=True)

    train_dataset = Dataset.from_generator(lambda: load_data_stream(source_path, target_path))
    print(len(train_dataset))
    # train_dataset = train_dataset.select(range(10000))
    tokenized_train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q","v"],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False,
        task_type=TaskType.SEQ_2_SEQ_LM
        )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # reporter = ['wandb'] if local_rank == 0 else "none"
    reporter = 'none'
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,

        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,

        # weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        # lr_scheduler_type="constant",
        save_total_limit=1,
        save_strategy="no",
        save_only_model=True,
        num_train_epochs=args.num_epochs,
        #evaluation_strategy="epoch",
        # predict_with_generate=True,
        fp16=False,
        bf16=True,
        report_to=reporter,
        logging_dir='/logs/',
        logging_steps=100,
        # gradient_accumulation_steps=4,
        dataloader_num_workers=10,
        seed=args.seed,
        # deepspeed = 'configs/t5_ds_config.json',
        # lr_scheduler_type="linear",
        # load_best_model_at_end=True,

    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    trainer.train()

    output_dir = os.path.join(output_dir, "final_ckpt")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()


import argparse
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA model weights with base T5 model.")
    parser.add_argument('--base_model_name', type=str, required=True, help="Base model name (e.g., google-t5/t5-3b).")
    parser.add_argument('--lora_model_path', type=str, required=True, help="Path to the LoRA fine-tuned model.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the merged model.")
    return parser.parse_args()

def main():
    args = parse_args()
    
  
    print(f"Loading base model: {args.base_model_name}")
    base_model = T5ForConditionalGeneration.from_pretrained(args.base_model_name)
    

    print(f"Loading LoRA model from: {args.lora_model_path}")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    

    print("Merging LoRA weights into base model...")
    lora_model.merge_and_unload()
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    state_dict = lora_model.base_model.state_dict()
    clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    torch.save(clean_state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
    print(f"Model weights saved to: {os.path.join(args.output_dir, 'pytorch_model.bin')}")
    

    tokenizer = T5Tokenizer.from_pretrained(args.lora_model_path)
    tokenizer.save_pretrained(args.output_dir)

    config = T5Config.from_pretrained(args.base_model_name)
    config.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

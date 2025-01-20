from huggingface_hub import HfApi, Repository, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import torch
import time

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# i = '@@'
# encoded_input = tokenizer(i, return_tensors='pt')
# print(encoded_input)
# print(tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True).strip())
# print(tokenizer.decode([29992], skip_special_tokens=True).strip())
# i = '||'
# encoded_input = tokenizer(i, return_tensors='pt')
# print(encoded_input)
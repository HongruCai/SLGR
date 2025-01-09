from huggingface_hub import HfApi, Repository, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
i = 'what is curosurf ?'
encoded_input = tokenizer(i, return_tensors='pt')
print(tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True).strip())

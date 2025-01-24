from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    DataCollatorWithPadding,
    GenerationConfig,
)
import torch
import os
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import json
from peft import PeftModel
from utils import T5Dataset, Trie, prefix_allowed_tokens_fn
import ujson
import sys
from accelerate import PartialState
from accelerate.utils import gather_object
import torch.distributed as dist
import argparse

class ValidationDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_source_len=64):

        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        self.queries = []
        with open(tsv_path, "r") as f:
            for line in f:
                query_id, query_text = line.strip().split("\t")
                self.queries.append(query_text)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        
        query_encodings = self.tokenizer(
            query,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        return {"input_ids": query_encodings.squeeze(), "labels": [0]}
    

def load_all_smtids(docid_to_smtid_path, tokenizer):

    with open(docid_to_smtid_path, "r") as f:
        docid_to_smtid = ujson.load(f)

    all_smtids = []
    for smtids in tqdm(docid_to_smtid.values(), desc="Building smtid list"):
        # if smtids[1:] not in all_smtids:
        labels = " ".join(f"c_{c}" for c in smtids[1:])
        labels = tokenizer.encode(labels)
        all_smtids.append(labels) 
    
    return all_smtids


def main(args):

    distributed_state = PartialState()


    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=torch.bfloat16, device_map=distributed_state.device)

    print("tokenizer loaded from " + args.model_path)
    print("model loaded from " + args.model_path)

    model.to(args.device)
    model.eval()

    dataset = ValidationDataset(tsv_path=args.dev_queries, tokenizer=tokenizer)
    print("Dataset loaded, size: ", len(dataset))
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=args.max_length
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator
    )


    code_list = load_all_smtids(args.docid_to_smtid, tokenizer)
    condidate_trie = Trie([[0] + x for x in code_list])
    print("\nTrie loaded, possible response count: ", len(condidate_trie))
    prefix_allowed_tokens = prefix_allowed_tokens_fn(condidate_trie)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating queries"):
            inputs = batch
            with distributed_state.split_between_processes(inputs['input_ids']) as splited_inputs:
                for splited_input in splited_inputs:
                    splited_input = splited_input.unsqueeze(0)
                    generation_config = GenerationConfig(
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=args.num_beams,
                        early_stopping=True,
                        use_cache=True,
                    )
                    batch_beams = model.generate(
                        input_ids=splited_input.to(distributed_state.device),
                        generation_config=generation_config,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    ).reshape(splited_input.shape[0], args.num_beams, -1)

                    for beams, input_ids in zip(batch_beams, splited_input):
                        rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
                        rank_list = [[int(item.split('_')[1]) for item in string.split()] for string in rank_list]
                        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)

                        result_entry = {
                            "input": input_text,
                            "predictions": rank_list,
                        }
                        results.append(result_entry)

    results = gather_object(results)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        ujson.dump(results, f)
    print(f"Results saved to {args.output_pat}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate T5 model with constraints.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--base_model_path', type=str, default='t5-small', help='Path to base model')
    parser.add_argument('--docid_to_smtid', type=str, default='../data/MSMARCO/filtered_docid_to_smtid.json', help='Path to docid_to_smtid.json')
    parser.add_argument('--dev_qrel', type=str, default='../data/MSMARCO/dev_qrel.json', help='Path to dev_qrel.json')
    parser.add_argument('--dev_queries', type=str, default='../data/MSMARCO/dev/raw.tsv', help='Path to dev queries TSV file')
    parser.add_argument('--output_path', type=str, default='result', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--num_beams', type=int, default=100, help='Number of beams for beam search')
    parser.add_argument('--max_new_tokens', type=int, default=36, help='Maximum number of tokens to generate')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum length for tokenized inputs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda", "cpu")')

    args = parser.parse_args()
    main(args)
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
        
        # 对 query 进行分词
        query_encodings = self.tokenizer(
            query,
            padding="max_length",
            max_length=self.max_source_len,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        return {"input_ids": query_encodings.squeeze(), "labels": [0]}
    

def load_all_smtids(docid_to_smtid_path, tokenizer):

    # 加载 docid_to_smtid 数据
    with open(docid_to_smtid_path, "r") as f:
        docid_to_smtid = ujson.load(f)
    
    # 提取所有 smtid
    all_smtids = []
    for smtids in tqdm(docid_to_smtid.values(), desc="Building smtid list"):
        # if smtids[1:] not in all_smtids:
        labels = " ".join(f"c_{c}" for c in smtids[1:])
        labels = tokenizer.encode(labels)
        all_smtids.append(labels)  # 添加去掉第一个元素的列表
    
    return all_smtids


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model_path = "./output/T5/t5-small"
    base_model_path = 't5-small'

    docid_to_smtid = '../data/MSMARCO/docid_to_smtid.json'
    dev_qrel = '../data/MSMARCO/dev_qrel.json'
    dev_queries = '../data/MSMARCO/dev/raw.tsv'


    distributed_state = PartialState()

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map=distributed_state.device)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_path, torch_dtype=torch.float16, device_map=distributed_state.device)
    
    print("tokenizer loaded from " + model_path)
    print("model loaded from " + model_path)

    model.to(device)
    model.eval()

    batch_size = 64
    num_beams = 100


    dataset = ValidationDataset(tsv_path=dev_queries, tokenizer=tokenizer)
    print("Dataset loaded, size: ", len(dataset))
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=64
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )


    code_list = ujson.load(open("../data/MSMARCO/t5_encoded_smtids.json"))
    code_list = code_list
    condidate_trie = Trie([[0] + x for x in code_list])
    print("\nTrie loaded, possible response count: ", len(condidate_trie))
    prefix_allowed_tokens = prefix_allowed_tokens_fn(condidate_trie)


    recall_count_at_1 = 0
    recall_count_at_5 = 0
    recall_count_at_10 = 0
    recall_count_at_20 = 0
    recall_count_at_100 = 0
    
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating queries"):
            inputs = batch
            local_results = []
            with distributed_state.split_between_processes(inputs['input_ids']) as splited_inputs:
                for splited_input in splited_inputs:
                    splited_input = splited_input.unsqueeze(0)
                    generation_config = GenerationConfig(
                        num_beams=num_beams,
                        max_new_tokens=36,
                        num_return_sequences=num_beams,
                        early_stopping=True,
                        use_cache=True,
                    )
                    batch_beams = model.generate(
                        input_ids=splited_input.to(distributed_state.device),
                        generation_config=generation_config,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    ).reshape(splited_input.shape[0], num_beams, -1)
                    for beams, input_ids in zip(
                        batch_beams, splited_input
                    ):

                        rank_list = tokenizer.batch_decode(
                            beams, skip_special_tokens=True)
                        rank_list = [ [int(item.split('_')[1]) for item in string.split()] for string in rank_list]
                        # print(rank_list)

                        input_text = tokenizer.decode(
                            input_ids, skip_special_tokens=True
                        ).strip()

                        result_entry = {
                            "input": input_text,
                            "predictions": rank_list,
                        }
                        local_results.append(result_entry)
    results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(results, local_results)  
    print("Results gathered, size: ", len(results))
    
    os.makedirs("result", exist_ok=True)
    with open("result/t5_small_res.json", "w") as f:
        ujson.dump(results, f)

    query_to_id = {}
    with open(dev_queries, "r") as f:
        for line in f:
            query_id, query_text = line.strip().split("\t")
            query_to_id[query_text] = query_id
    
    with open(dev_qrel, "r") as f:
        qid_to_docid = ujson.load(f)
    
    with open(docid_to_smtid, "r") as f:
        docid_to_smtid = ujson.load(f)
    
    for result in results:
        query = result["input"].strip()
        predictions = result["predictions"]
        qid = query_to_id[query]
        docids = qid_to_docid[qid]
        posids =[docid_to_smtid[docid][1:] for docid in docids]
        hits = [i for i, x in enumerate(predictions) if x in posids]
        if len(hits) != 0:
            recall_count_at_100 += 1
            if hits[0] < 20:
                recall_count_at_20 += 1
            if hits[0] < 10:
                recall_count_at_10 += 1
            if hits[0] < 5:
                recall_count_at_5 += 1
            if hits[0] == 0:
                recall_count_at_1 += 1

    distributed_state.print("Total queries: ", len(results))
    distributed_state.print("Recall@1: ", recall_count_at_1 / len(results))
    distributed_state.print("Recall@5: ", recall_count_at_5 / len(results))
    distributed_state.print("Recall@10: ", recall_count_at_10 / len(results))
    distributed_state.print("Recall@20: ", recall_count_at_20 / len(results))
    distributed_state.print("Recall@100: ", recall_count_at_100 / len(results))



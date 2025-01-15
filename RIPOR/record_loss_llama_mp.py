import random
import csv
from more_itertools import chunked
import ast
from seal.retrieval import SEALSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
import sys
from seal.index import FMIndex
import json
from fuzzywuzzy import fuzz
import math
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import nltk
from tqdm import tqdm
import torch
from peft import PeftModel
import argparse
import os
from torch import nn
import concurrent.futures
import ujson
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_model_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("BF16 supported, loading model in BF16...")
        return torch.bfloat16
    else:
        print("BF16 not supported, falling back to FP16...")
        return torch.float16

def load_dev_queries(file_path):

    queries = {}
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            query_id, query_text = row
            queries[query_id] = query_text
    return queries
  
def compute_loss(model, tokenizer, inputs, ids, device='cuda'):
    # 将 inputs 和 ids 拼接成一个完整的序列
    input_text = inputs[0]  # 单个输入文本
    ids_text = " ".join(f"c_{c}" for c in ids)
    full_text = f"{input_text} {ids_text} </s>"

    # 编码输入
    encodings = tokenizer(
        full_text,
        padding="max_length",
        max_length=128,  # 根据具体任务调整
        truncation=True,
        return_tensors="pt"
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 构造标签，将 query 部分 mask 掉
    labels = input_ids.clone()
    query_len = len(tokenizer(input_text, truncation=True, return_tensors="pt")["input_ids"][0])
    labels[:, :query_len] = -100  # mask 掉 query 部分
    labels[attention_mask == 0] = -100

    # print(labels)
    # print(input_ids)
    # 计算 loss
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    return outputs.loss.detach().cpu().item()


def compute_loss_batch(model, tokenizer, inputs, ids_list, device='cuda'):

    # 将 inputs 和 ids_list 拼接成完整序列
    full_texts = []
    for input_text, ids in zip(inputs, ids_list):
        ids_text = " ".join(f"c_{c}" for c in ids)
        full_texts.append(f"{input_text} {ids_text} </s>")

    # 编码输入
    encodings = tokenizer(
        full_texts,
        padding="max_length",
        max_length=128,  # 根据具体任务调整
        truncation=True,
        return_tensors="pt"
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 构造标签，将 query 部分 mask 掉
    labels = input_ids.clone()
    for i, input_text in enumerate(inputs):
        query_len = len(tokenizer(input_text, truncation=True, return_tensors="pt")["input_ids"][0])
        labels[i, :query_len] = -100  # mask 掉 query 部分
    labels[attention_mask == 0] = -100

    # 计算 loss
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    return outputs.loss.detach().cpu().item()


def process_query(model, tokenizer, docid_to_smtids, qid, query, query2doc_pos, all_docids, neg_nums, batch_size=32):
    """
    处理单个 query 的正负样本 loss 计算，支持负样本批量计算。
    """
    pos_docs = query2doc_pos.get(qid, [])
    docs = list(all_docids - set(pos_docs))
    neg_docs = random.sample(docs, neg_nums)

    # 获取正负样本的 SMT IDs
    pos_ids = [docid_to_smtids[docid][1:] for docid in pos_docs]
    neg_ids = [docid_to_smtids[docid][1:] for docid in neg_docs]
    input_text = query

    # 正样本 loss 计算
    pos_losses = []
    for pos_input, pos_id in zip(len(pos_docs) * [input_text], pos_ids):
        loss = compute_loss(model, tokenizer, [pos_input], pos_id)
        pos_losses.append(loss)

    # 负样本 loss 计算（分批处理）
    neg_losses = []
    for i in range(0, len(neg_ids), batch_size):
        batch_neg_ids = neg_ids[i:i + batch_size]
        batch_input_texts = len(batch_neg_ids) * [input_text]
        loss = compute_loss_batch(model, tokenizer, batch_input_texts, batch_neg_ids)
        neg_losses.append(loss)

    return qid, {"pos": pos_losses, "neg": neg_losses}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMa with FM Index")
    parser.add_argument('--docid_to_smtid_path', type=str, default='../data/MSMARCO/filtered_docid_to_smtid.json', help='Path to filtered docid_to_smtid.json')
    parser.add_argument('--query_to_doc_path', type=str, default='../data/MSMARCO/dev_qrel.json', help='Path to dev_qrel.json')
    parser.add_argument('--dev_query_path', type=str, default='../data/MSMARCO/dev/raw.tsv', help='Path to raw.tsv for dev queries')
    parser.add_argument('--neg_nums', type=int, default=256, help="Number of negative samples per query")
    parser.add_argument('--model_path', type=str, default='data/llama1_fm_index', help="Path to the PEFT model")
    parser.add_argument('--base_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Base path for the Llama model")
    parser.add_argument('--output_file', type=str, default='result/llama_7b_loss.json', help="Path to save the results")
    parser.add_argument('--num_processes', type=int, default=10, help="Number of processes to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for loss computation")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.docid_to_smtid_path, 'r', encoding='utf-8') as fin:
        docid_to_smtids = ujson.load(fin)
    all_docids = set(list(docid_to_smtids.keys()))

    with open(args.query_to_doc_path, 'r', encoding='utf-8') as fin:
        query2doc = ujson.load(fin)

    query2doc_pos = {}
    for qid, q in query2doc.items():
        query2doc_pos[qid] = []
        for docid, score in q.items():
            query2doc_pos[qid].append(docid)

    queries = load_dev_queries(args.dev_query_path)

    model_path = args.model_path
    base_path = args.base_path
    
    # Get model precision
    precision = get_model_precision()

    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    base_model = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=precision, device_map='auto')
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
                base_model,
                model_path,
                torch_dtype=precision,
                 device_map='auto'
            )

    model = model.to("cuda")
    model.eval()
    print('Model loaded from:', model_path)

    rec_loss = {}
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
            futures = []
            total_queries = len(queries)

            with tqdm(total=total_queries, desc="Processing queries") as pbar:
                for qid, query in queries.items():
                    # 将查询任务提交到线程池，包含 docid 的选择逻辑
                    future = executor.submit(
                        process_query, model, tokenizer, docid_to_smtids, qid, query, query2doc_pos, all_docids, args.neg_nums, args.batch_size
                    )
                    futures.append(future)

                # 获取任务结果
                for future in as_completed(futures):
                    qid, result = future.result()
                    rec_loss[qid] = result
                    pbar.update(1)  # 更新进度条

                    # 保存中间结果
                    with open(args.output_file, 'w') as fout:
                        json.dump(rec_loss, fout)

    # Final save
    with open(args.output_file, 'w') as fout:
        json.dump(rec_loss, fout)

    print(f"Losses saved to {args.output_file}")

if __name__ == "__main__":
    main()
        


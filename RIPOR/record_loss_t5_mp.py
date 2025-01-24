import ujson
import sys
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
import torch
import csv
import random
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import json
import argparse
from peft import PeftModel
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


    input_encodings = tokenizer(inputs,
                                padding="max_length",
                                max_length=64,
                                truncation=True,
                                return_tensors="pt")
    

    ids = " ".join(f"c_{c}" for c in ids)

    labels_encodings = tokenizer(
            ids,
            padding="max_length",
            max_length=33,
            truncation=True,
            return_tensors="pt",
        )

    with torch.no_grad():
        outputs = model(
            input_ids=input_encodings["input_ids"].to(device),
            labels=labels_encodings["input_ids"].to(device),
        )

    return outputs.loss.detach().cpu().item()


def compute_loss_batch(model, tokenizer, inputs, ids_list, device='cuda'):

    input_encodings = tokenizer(
        inputs,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )

    labels_list = [" ".join(f"c_{c}" for c in ids) for ids in ids_list]
    labels_encodings = tokenizer(
        labels_list,
        padding="max_length",
        max_length=33,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_encodings["input_ids"].to(device),
            labels=labels_encodings["input_ids"].to(device),
        )

    return outputs.loss.detach().cpu().item()


def process_query(model, tokenizer, docid_to_smtids, qid, query, query2doc_pos, all_docids, neg_nums, batch_size=32):

    pos_docs = query2doc_pos.get(qid, [])
    docs = list(all_docids - set(pos_docs))
    neg_docs = random.sample(docs, neg_nums)


    pos_ids = [docid_to_smtids[docid][1:] for docid in pos_docs]
    neg_ids = [docid_to_smtids[docid][1:] for docid in neg_docs]
    input_text = query


    pos_losses = []
    for pos_input, pos_id in zip(len(pos_docs) * [input_text], pos_ids):
        loss = compute_loss(model, tokenizer, [pos_input], pos_id)
        pos_losses.append(loss)


    neg_losses = []
    for i in range(0, len(neg_ids), batch_size):
        batch_neg_ids = neg_ids[i:i + batch_size]
        batch_input_texts = len(batch_neg_ids) * [input_text]
        loss = compute_loss_batch(model, tokenizer, batch_input_texts, batch_neg_ids)
        neg_losses.append(loss)

    return qid, {"pos": pos_losses, "neg": neg_losses}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for calculating positive and negative losses.")
    parser.add_argument('--docid_to_smtid_path', type=str, default='../data/MSMARCO/filtered_docid_to_smtid.json', help='Path to filtered docid_to_smtid.json')
    parser.add_argument('--query_to_doc_path', type=str, default='../data/MSMARCO/dev_qrel.json', help='Path to dev_qrel.json')
    parser.add_argument('--dev_query_path', type=str, default='../data/MSMARCO/dev/raw.tsv', help='Path to raw.tsv for dev queries')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')  
    parser.add_argument('--base_model_path', type=str, default='t5-base', help='Path to pretrained model')
    parser.add_argument('--output_path', type=str, default='result/large_loss.json', help='Path to save loss results')
    parser.add_argument('--neg_nums', type=int, default=256, help='Number of negative samples per query')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for negative samples')

    args = parser.parse_args()

    # Load data
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
    precision = get_model_precision()
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(args.base_model_path, torch_dtype=precision)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=precision).to('cuda')
    model.eval()

    print("Model loaded from", args.model_path)
    
    rec_loss = {}

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        total_queries = len(queries)

        with tqdm(total=total_queries, desc="Processing queries") as pbar:
            for qid, query in queries.items():
                
                future = executor.submit(
                    process_query, model, tokenizer, docid_to_smtids, qid, query, query2doc_pos, all_docids, args.neg_nums, args.batch_size
                )
                futures.append(future)


            for future in as_completed(futures):
                qid, result = future.result()
                rec_loss[qid] = result
                pbar.update(1)  


                with open(args.output_path, 'w') as fout:
                    json.dump(rec_loss, fout)

    # Final save
    with open(args.output_path, 'w') as fout:
        json.dump(rec_loss, fout)

    print(f"Losses saved to {args.output_path}")

            
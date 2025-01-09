import ujson
import sys
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
import torch
import csv
import random
from transformers import AutoTokenizer, T5ForConditionalGeneration
import json


def ddp_setup():
    init_process_group(backend="nccl")

def load_dev_queries(file_path):

    queries = {}
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            query_id, query_text = row
            queries[query_id] = query_text
    return queries

def compute_loss(model, tokenizer, inputs, ids, device='cuda'):

    # Tokenize 输入和输出
    input_encodings = tokenizer(inputs,
                                add_special_tokens=True,
                                padding="longest",  # pad to max sequence length in batch
                                truncation="longest_first",  # truncates to self.max_length
                                max_length=256,
                                return_attention_mask=True)
    input_encodings = {k: torch.tensor(v).to(device) for k, v in input_encodings.items()}
    
    # 禁用梯度计算
    ids = torch.tensor(ids).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_encodings["input_ids"],
            attention_mask=input_encodings["attention_mask"],
            labels=ids,
        )
    # 返回每个样本的损失值
    return outputs.loss.detach().cpu().item()

if '__main__' in __name__:
    docid_to_smtid_path = 'experiments-full-t5seq-aq/t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json'
    q2doc = 'data/msmarco-full/dev_qrel.json'
    dev_qery_path = 'data/msmarco-full/dev_queries/raw.tsv'

    with open(docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin) # docid: from 0 to 8841822
    all_docids = list(docid_to_smtids.keys())

    with open(q2doc) as fin:
        query2doc = ujson.load(fin)
    
    query2doc_pos = {}
    for qid, q in query2doc.items():
        for docid, score in q.items():
            if score > 0:
                if qid not in query2doc_pos:
                    query2doc_pos[qid] = []
                query2doc_pos[qid].append(docid)

    queries = load_dev_queries(dev_qery_path)

    neg_nums = 256
    # ddp_setup()
    pretrained_path = 'experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_0_large/checkpoint'
    model = T5ForConditionalGeneration.from_pretrained(pretrained_path).to('cuda')
    # model = T5ForDocIDGeneration.from_pretrained(pretrained_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model.eval()

    rec_loss = {}
    with torch.no_grad():
        for qid, query in tqdm(queries.items()):
            pos_docs = query2doc_pos[qid]
            neg_docs = list(set(all_docids) - set(pos_docs))
            neg_docs = random.sample(neg_docs, neg_nums)
            
            pos_ids = [docid_to_smtids[docid][1:] for docid in pos_docs]
            neg_ids = [docid_to_smtids[docid][1:] for docid in neg_docs]
            input_text = 'query: ' + query

            pos_losses = []
            for pos_input, pos_id in zip(len(pos_docs) * [input_text], pos_ids):
                loss = compute_loss(model, tokenizer, [pos_input], [pos_id])
                pos_losses.append(loss)  # 取单个样本的 loss

            # 为每个负样本单独计算 loss
            neg_losses = []
            for neg_input, neg_id in zip(len(neg_docs) * [input_text], neg_ids):
                loss = compute_loss(model, tokenizer, [neg_input], [neg_id])
                neg_losses.append(loss)  # 取单个样本的 loss
            # print(pos_losses, neg_losses)
            rec_loss[qid] = {
                "pos": pos_losses,
                "neg": neg_losses
            }
            with open('result/large_loss.json', 'w') as fout:
                json.dump(rec_loss, fout)
        with open('result/large_loss.json', 'w') as fout:
            json.dump(rec_loss, fout)

            
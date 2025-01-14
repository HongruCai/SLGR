import ujson
import sys
from tqdm import tqdm
import csv
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

recall_count_at_1 = 0
recall_count_at_5 = 0
recall_count_at_10 = 0
recall_count_at_20 = 0
recall_count_at_100 = 0

with open('result/t5_debug.json') as fin:
    results = ujson.load(fin)

docid_to_smtid = '../data/MSMARCO/filtered_docid_to_smtid.json'
dev_qrel = '../data/MSMARCO/dev_qrel.json'
dev_queries = '../data/MSMARCO/dev/raw.tsv'

print(len(results))

query_to_id = {}
with open(dev_queries, "r") as f:
    reader = csv.reader(f, delimiter='\t')  
    for row in reader:
        key = row[0]
        value = row[1:]
        value = value[0].strip()
        value = tokenizer.encode(value, max_length=64, truncation=True)
        value = '_'.join([str(x) for x in value])
        # print(value)
        query_to_id[value] = key

with open(dev_qrel, "r") as f:
    qid_to_docid = ujson.load(f)

with open(docid_to_smtid, "r") as f:
    docid_to_smtid = ujson.load(f)

for result in tqdm(results):
    query = result["input"].strip()
    predictions = result["predictions"]
    assert len(predictions[0]) == 32
    query = tokenizer.encode(query, max_length=64, truncation=True)
    query = '_'.join([str(x) for x in query])
    if query not in query_to_id:
        print("Query not found: ", query)
        continue
    qid = query_to_id[query]
    docids = qid_to_docid[qid]

    posids =[docid_to_smtid[docid][1:] for docid in docids]
    assert len(posids[0]) == 32

    # print(predictions)
    hits = [i for i, x in enumerate(predictions) if x in posids]
    # print(hits)
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

print("Total queries: ", len(results))
print("Recall@1: ", recall_count_at_1 / len(results))
print("Recall@5: ", recall_count_at_5 / len(results))
print("Recall@10: ", recall_count_at_10 / len(results))
print("Recall@20: ", recall_count_at_20 / len(results))
print("Recall@100: ", recall_count_at_100 / len(results))
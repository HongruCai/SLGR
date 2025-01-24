import random
import csv
from more_itertools import chunked
import ast
from seal.retrieval import SEALSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
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
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


nltk.download('stopwords')
banned = set(stopwords.words('english'))
def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)

def extract_spans(text, source, n_samples, min_length, max_length, temperature=1.0):
    source = source.split("||", 1)[0]
    query_tokens = source.split()
    query_tokens_lower = [t.lower() for t in query_tokens]

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens]

    matches = defaultdict(int)

    for i1, _ in enumerate(query_tokens_lower):
        j1 = i1+3
        str_1 = " ".join(query_tokens_lower[i1:j1])

        for (i2, j2) in span_iterator(passage_tokens_lower, 3):
            str_2 = " ".join(passage_tokens_lower[i2:j2])
            ratio = fuzz.ratio(str_1, str_2) / 100.0
            matches[i2] += ratio

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1])))
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            weights = [math.exp(float(w) / temperature) for w in weights]
            Z = sum(weights)
            weights = [w / Z for w in weights]

        indices = random.choices(indices, weights=weights, k=n_samples)

    span_list=[]
    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        span_list.append(span)
    return span_list

def extract_parts(doc, q):
    parts = doc.split("<extra_id_99>", 1)
    title = parts[0].strip()

    remaining_text = parts[1].strip()
    text_parts = remaining_text.split("<extra_id_97>")

    text = text_parts[0].strip()
    
    queries = [q.split("<extra_id_99>")[0].strip() for q in text_parts[1:] if "<extra_id_99>" in q]
    bodies = extract_spans(text, q, 10, 10, 10, temperature=1.0)
    
    return title, bodies, queries

def get_doc_identifier(fm_index, tokenizer, answer):
    tkn_a = tokenizer(' ' + answer)
    pos_rg = fm_index.get_range(tkn_a['input_ids'][:-1])
    pos_tkn = fm_index.locate(pos_rg[0])
    pos_doc = fm_index.get_doc_index(pos_tkn)
    return pos_doc

def get_model_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("BF16 supported, loading model in BF16...")
        return torch.bfloat16
    else:
        print("BF16 not supported, falling back to FP16...")
        return torch.float16
    
def prepare_batch_inputs_outputs(query, doc_identifiers):
    inputs = []
    labels = []

    for i in range(3):
        title_input = f"{query} || title || +"
        title_label = doc_identifiers['title']
        inputs.append(title_input)
        labels.append(title_label)

    for text in doc_identifiers['text']:
        body_input = f"{query} || body || +"
        inputs.append(body_input)
        labels.append(text)

    for pseudo_query in doc_identifiers['queries']:
        pseudo_input = f"{query} || query || +"
        inputs.append(pseudo_input)
        labels.append(pseudo_query)


    return inputs, labels


def compute_doc_loss(model, tokenizer, query, doc_identifiers):

    inputs, labels = prepare_batch_inputs_outputs(query, doc_identifiers)

    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
    label_encodings = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**input_encodings, labels=label_encodings["input_ids"])

    avg_loss = outputs.loss.detach().cpu().item()

    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Compute document losses with T5 model.")
    parser.add_argument('--index', type=str, required=True, help="Path to the FM Index file.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file with queries and answers.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned T5 model.")
    parser.add_argument('--output_file', type=str, default='result/t5_loss.json', help="Path to save the results.")
    parser.add_argument('--total_docs', type=int, default=21015324, help="Total number of documents in the index.")
    parser.add_argument('--num_neg_samples', type=int, default=256, help="Number of negative samples per query.")
    parser.add_argument('--num_processes', type=int, default=1, help="Number of processes to use.")
    return parser.parse_args()

def process_query(model, tokenizer, fm_index, q, answers, total_docs, num_neg_samples):
    positive_idens = []
    negative_idens = []
    poditive_docs = []

    for a in answers:
        tkn_a = tokenizer(' ' + a)
        pos_rg = fm_index.get_range(tkn_a['input_ids'][:-1])
        pos_tkn = fm_index.locate(pos_rg[0])
        pos_doc = fm_index.get_doc_index(pos_tkn)
        doc_content = fm_index.get_doc(pos_doc)
        doc_content = tokenizer.decode(doc_content)
        title, text, queries = extract_parts(doc_content, q)
        positive_idens.append({'title': title, 'text': text, 'queries': queries})
        poditive_docs.append(pos_doc)

    available_neg_docs = list(set(range(total_docs)) - set(poditive_docs))
    neg_docs = random.sample(available_neg_docs, num_neg_samples)
    for neg_doc in neg_docs:
        doc_content = fm_index.get_doc(neg_doc)
        doc_content = tokenizer.decode(doc_content)
        title, text, queries = extract_parts(doc_content, q)
        negative_idens.append({'title': title, 'text': text, 'queries': queries})


    pos_loss = [compute_doc_loss(model, tokenizer, q, pos) for pos in positive_idens]
    neg_loss = [compute_doc_loss(model, tokenizer, q, neg) for neg in negative_idens]

    return q, {'pos_loss': pos_loss, 'neg_loss': neg_loss}


def main():
    args = parse_args()
    

    print('Loading index...')
    fm_index = FMIndex.load(args.index)
    print('Index loaded')
    print('Index size:', fm_index.n_docs)
    

    data_mapping = {}
    with open(args.csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for query, answers in reader:
            parsed_answers = ast.literal_eval(answers)
            data_mapping[query] = parsed_answers
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)


    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    precision = get_model_precision()
    model = T5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=precision).to('cuda')
    print('Model loaded from:', args.model_path)
    
    rec = {}
    

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        total_queries = len(data_mapping)
        
        with tqdm(total=total_queries, desc="Processing queries") as pbar:
            for q, answers in data_mapping.items():
                futures.append(executor.submit(process_query, model, tokenizer, fm_index, q, answers, args.total_docs, args.num_neg_samples))
            
            for future in as_completed(futures):
                q, result = future.result()
                rec[q] = result
                

                pbar.update(1)


                with open(args.output_file, 'w') as f_out:
                    json.dump(rec, f_out)
    
    with open(args.output_file, 'w') as f_out:
        json.dump(rec, f_out)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
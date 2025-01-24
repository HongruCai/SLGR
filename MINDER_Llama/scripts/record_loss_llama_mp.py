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

    parts = doc.split("@@", 1)
    title = parts[0].strip()
    remaining_text = parts[1].strip()
    text_parts = remaining_text.split("||")

    text = text_parts[0].strip()
    
    queries = [q.split("@@")[0].strip() for q in text_parts[1:] if "@@" in q]
    bodies = extract_spans(text, q, 10, 10, 10, temperature=1.0)
    
    return  bodies

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
    weights = []

    for text in doc_identifiers['text']:
        body_input = f"{query}"
        inputs.append(body_input)
        labels.append(text)
        weights.append(1)

    return inputs, labels, weights


  
def compute_doc_loss_llama(model, tokenizer, query, doc_identifiers):
    inputs, labels, weights = prepare_batch_inputs_outputs(query, doc_identifiers)

    full_texts = [inp + " " + lbl for inp, lbl in zip(inputs, labels)]

    encodings = tokenizer(full_texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to("cuda")

    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=256).to("cuda")
    input_lengths = input_encodings["input_ids"].shape[1]

    labels = encodings["input_ids"].clone()
    labels[:, :input_lengths] = -100  
    # print(labels.shape)
    # print(encodings["input_ids"].shape)
    with torch.no_grad():
        outputs = model(input_ids=encodings["input_ids"], labels=labels)

    loss_per_sample = outputs.loss.detach().cpu().item()

    avg_loss = (loss_per_sample * torch.tensor(weights, dtype=torch.float32).to("cuda")).mean().item()

    return avg_loss

def compute_losses_for_query(model, tokenizer, fm_index, q, data_mapping, args):
    positive_idens = []
    negative_idens = []
    poditive_docs = []
    answers = data_mapping[q]
    
    for a in answers:
        tkn_a = tokenizer(' ' + a)
        pos_rg = fm_index.get_range(tkn_a['input_ids'][:-1])
        pos_tkn = fm_index.locate(pos_rg[0])
        pos_doc = fm_index.get_doc_index(pos_tkn)
        doc_content = fm_index.get_doc(pos_doc)
        doc_content = tokenizer.decode(doc_content)
        text = extract_parts(doc_content, q)
        positive_idens.append({'text': text})
        poditive_docs.append(pos_doc)

    # Sample negative documents
    available_neg_docs = list(set(range(args.total_docs)) - set(poditive_docs))
    neg_docs = random.sample(available_neg_docs, args.num_neg_samples)
    
    for neg_doc in neg_docs:
        doc_content = fm_index.get_doc(neg_doc)
        doc_content = tokenizer.decode(doc_content)
        text = extract_parts(doc_content, q)
        negative_idens.append({'text': text})

    # Compute losses for positive and negative samples
    pos_loss = []
    neg_loss = []
    
    for pos in positive_idens:
        pos_loss.append(compute_doc_loss_llama(model, tokenizer, q, pos))
    for neg in negative_idens:
        neg_loss.append(compute_doc_loss_llama(model, tokenizer, q, neg))
    
    return q, {'pos_loss': pos_loss, 'neg_loss': neg_loss}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMa with FM Index")
    parser.add_argument('--index', type=str, default='../data/FMIndex/llama_psgs_w100.fm_index', help="Path to the FM Index file")
    parser.add_argument('--csv_file', type=str, default='../data/NQ/nq-test.csv', help="Path to the CSV file with queries and answers")
    parser.add_argument('--model_path', type=str, default='data/llama1_fm_index', help="Path to the PEFT model")
    parser.add_argument('--base_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Base path for the Llama model")
    parser.add_argument('--output_file', type=str, default='result/llama_7b_loss.json', help="Path to save the results")
    parser.add_argument('--total_docs', type=int, default=21015324, help="Total number of documents in the index")
    parser.add_argument('--num_neg_samples', type=int, default=256, help="Number of negative samples per query")
    parser.add_argument('--num_processes', type=int, default=10, help="Number of processes to use")
    return parser.parse_args()

def main():
    args = parse_args()

    print('Loading index...')
    fm_index = FMIndex.load(args.index)
    print('Index loaded')
    print('Index size:', fm_index.n_docs)

    rec = {}
    data_mapping = {}

    with open(args.csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for query, answers in reader:
            parsed_answers = ast.literal_eval(answers)
            data_mapping[query] = parsed_answers

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    model_path = args.model_path
    base_path = args.base_path
    
    # Get model precision
    precision = get_model_precision()

    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(base_path, torch_dtype=precision, device_map='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # model = PeftModel.from_pretrained(
    #             model,
    #             model_path,
    #             torch_dtype=precision,
    #             device_map='auto'
    #         )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    # model = model.to("cuda")
    print('Model loaded from:', model_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        
        total_queries = len(data_mapping)
        with tqdm(total=total_queries, desc="Processing queries") as pbar:

            def update_progress(future):
                pbar.update(1)  

            for q in data_mapping.keys():
                futures.append(executor.submit(compute_losses_for_query, model, tokenizer, fm_index, q, data_mapping, args))
            
            for future in concurrent.futures.as_completed(futures):
                future.add_done_callback(update_progress)
                
                q, loss_data = future.result()
                rec[q] = loss_data

                # Save intermediate results
                with open(args.output_file, 'w') as f_out:
                    json.dump(rec, f_out)

    # Final result
    with open(args.output_file, 'w') as f_out:
        json.dump(rec, f_out)

if __name__ == "__main__":
    main()
        


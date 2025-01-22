
import argparse

import math
import numpy as np
import random
import json
from tqdm import tqdm



# MINDER T5
# eval_failes = ['result/MINDER/t5/t5_small_loss.json', 'result/MINDER/t5/t5_base_loss.json', 
#                'result/MINDER/t5/t5_large_loss.json', 'result/MINDER/t5/t5_3b_loss.json',
#                'result/MINDER/t5/t5_11b_loss.json']

# MINDER LLAMA
# eval_failes = ['result/MINDER/llama/llama_7b_loss.json', 'result/MINDER/llama/llama_13b_loss.json', 'result/MINDER/llama/llama_70b_loss.json']

# MINER LLAMA Data
# eval_failes = ['result/MINDER/llama/llama_7b02_loss.json', 'result/MINDER/llama/llama_7b04_loss.json', 'result/MINDER/llama/llama_7b06_loss.json',
#                'result/MINDER/llama/llama_7b08_loss.json', 'result/MINDER/llama/llama_7b_loss.json']

# RIPOR T5
eval_failes = ['result/RIPOR/t5/rt5_small_loss.json', 'result/RIPOR/t5/rt5_base_loss.json', 
               'result/RIPOR/t5/rt5_large_loss.json', 'result/RIPOR/t5/rt5_3b_loss.json',
               'result/RIPOR/t5/rt5_11b_loss.json',]

all_res = {}
for eval_file in tqdm(eval_failes):
    query_scores = []
    # print('Evaluate on:', eval_file)

    neg_num = 256


    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for query in data:
            # pos_loss = data[query]['pos_loss']
            # neg_loss = data[query]['neg_loss']
            pos_loss = data[query]['pos']
            neg_loss = data[query]['neg']
            avg_neg_loss = sum(neg_loss) / len(neg_loss)
            neg_loss_sum = avg_neg_loss * neg_num
                
            pos_loss = min(pos_loss)
            
            # method 1
            neg_loss = neg_loss_sum
            s = neg_loss/(neg_loss+pos_loss)
            s = -np.log(s)

            # method 2

            # s = pos_loss/(neg_loss+pos_loss)

            # method 3
            # s = pos_loss

            # method 4
            # pos_loss = np.exp(-pos_loss)
            # neg_loss = sum([np.exp(-l) for l in neg_loss])
            # s = pos_loss/(pos_loss+neg_loss)
            # s = -np.log(s)

            # method 5

            query_scores.append(s)
    all_res[eval_file] = {'queries': len(query_scores), 'scores': sum(query_scores) / len(query_scores)}

max_file_length = max(len(file) for file in all_res)

# Calculate the max length for queries
max_queries_length = max(len(str(result['queries'])) for result in all_res.values())

# Print results with alignment
for file, result in all_res.items():
    print(f"{file.ljust(max_file_length)}: queries={str(result['queries']).rjust(max_queries_length)}, scores={result['scores']}")


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
eval_failes = ['result/MINDER/llama/llama_7b02_loss.json', 'result/MINDER/llama/llama_7b04_loss.json', 'result/MINDER/llama/llama_7b06_loss.json',
               'result/MINDER/llama/llama_7b08_loss.json', 'result/MINDER/llama/llama_7b_loss.json']
all_res = {}
for eval_file in tqdm(eval_failes):
    query_scores = []
    # print('Evaluate on:', eval_file)

    neg_num = 256


    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for query in data:
            pos_loss = data[query]['pos_loss']
            neg_loss = data[query]['neg_loss']
            assert len(neg_loss) == 256
            pos_loss = min(pos_loss)
            
            # method 1
            # neg_loss = sum(neg_loss)
            # s = neg_loss/(neg_loss+pos_loss)
            # s = -np.log(s)

            # method 2
            # neg_loss = sum(neg_loss)
            # s = pos_loss/(neg_loss+pos_loss)

            # method 3
            # s = pos_loss

            # method 4
            pos_loss = np.exp(-pos_loss)
            neg_loss = sum([np.exp(-l) for l in neg_loss])
            s = pos_loss/(pos_loss+neg_loss)
            s = -np.log(s)

            query_scores.append(s)
    all_res[eval_file] = {'queries': len(query_scores), 'scores': sum(query_scores) / len(query_scores)}

for eval_file in all_res:
    print(eval_file, all_res[eval_file])

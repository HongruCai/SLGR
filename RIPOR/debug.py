import json
import ujson
from tqdm import tqdm


example_path = '../data/MSMARCO/query_to_docid.train.json'
docid_to_smtid_path = '../data/MSMARCO/docid_to_smtid.json'


# with open(docid_to_smtid_path) as fin:
#     docid_to_smtid = ujson.load(fin)

# examples = []
# with open(example_path) as fin:
#     for i, line in tqdm(enumerate(fin)):
#         example = ujson.loads(line)
#         docid, query = example["docid"], example["query"]
#         smtid = docid_to_smtid[docid]
#         examples.append((query, smtid))
# print(examples[0])

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
extra_tokens = []
for count in range(256):
    extra_tokens.append('c_'+str(count))
print('number of extra tokens: ', len(extra_tokens))
tokenizer.add_tokens(extra_tokens)

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
        # print(labels)
        all_smtids.append(labels)  # 添加去掉第一个元素的列表
    
    return all_smtids

all_smtids = load_all_smtids(docid_to_smtid_path, tokenizer)
with open("../data/MSMARCO/t5_encoded_smtids.json", "w") as f:
    ujson.dump(all_smtids, f)
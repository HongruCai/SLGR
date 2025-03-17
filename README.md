
# Exploring Training and Inference Scaling Laws in Generative Retrieval

## Overview

Generative retrieval has emerged as a novel paradigm that leverages large language models (LLMs) to autoregressively generate document identifiers. Although promising, the mechanisms that underpin its performance and scalability remain largely unclear. We conduct a systematic investigation of **training and inference scaling laws** in generative retrieval, exploring how model size, training data scale, and inference-time compute jointly influence retrieval performance. To address the lack of suitable metrics, we propose a novel evaluation measure inspired by contrastive entropy and generation loss, providing a continuous performance signal that enables robust comparisons across diverse generative retrieval methods. Our experiments show that n-gram-based methods demonstrate strong alignment with both training and inference scaling laws, especially when paired with larger LLMs. Furthermore, increasing inference computation yields substantial performance gains, revealing that generative retrieval can significantly benefit from higher compute budgets at inference. Across these settings, LLaMA models consistently outperform T5 models, suggesting a particular advantage for larger decoder-only models in generative retrieval. Taken together, our findings underscore that model sizes, data availability, and inference computation interact to unlock the full potential of generative retrieval, offering new insights for designing and optimizing future systems.

## Requirements

We need two different environments to run the experiments: 

For MINDER_Llama and RIPOR:

```bash
cd MINDER_LLaMA 
conda env create -f environment.yaml
conda activate mllama
```

For MINDER_T5:

```bash
cd MINDER_T5
conda env create -f environment.yaml
conda activate mt5
```

## Data

We use NQ dataset for MINER experiments and MSMARCO for RIPOR experiments.
The preprocessed data and FMIndex can be downloaded from [Huggingface](https://huggingface.co/datasets/HenryCai/SLGR_data), and you can put the data in the `data` folder.
The FMIndex should work well if the environment is set up correctly, but we suggest re-building the FMIndex in your environment.

## Experiments

### MINDER

[MINDER](https://arxiv.org/abs/2305.16675) is a generative retrieval method that leverages text spans (body text, title, and pseudo-query) as document identifiers. For simplicity, we only use the body text as the document identifier. 

#### MINDER_LLaMA
1. Install FMIndex:

Follow the instructions in the [SEAL](https://github.com/facebookresearch/SEAL). You may need to clone the SEAL repository to install the sdsl-lite.

```bash
cd MINDER_LLaMA 
conda activate mllama
# install FMIndex
```

2. Data preparation:

We use the Natural Questions dataset. You can use `scripts/llama_index.sh` to build the FMIndex.

3. Run the experiments

```bash
# train
bash scripts/finetune_llama.sh
# test if you need
bash scripts/test_llama.sh
# eval loss
bash scripts/eval_loss.sh
```

#### MINDER_T5

1. Install FMIndex:

The steps are the same as MINDER_LLaMA, but we need another environment.

```bash
cd MINDER_T5
conda activate mt5
# install FMIndex
```

2. Data preparation:

We use the Natural Questions dataset. You can use `scripts/t5_index.sh` to build the FMIndex.

3. Run the experiments

```bash
# train
bash scripts/train.sh
# test if you need
bash scripts/test_t5.sh
# eval loss
bash scripts/eval_loss.sh
```

### RIPOR

[RIPOR](https://arxiv.org/abs/2311.09134) is a generative retrieval method that leverages codebooks to learn discrete representations of documents. We directly use the data provided by the authors.

1. Environment

```bash
cd RIPOR
conda activate mllama
```

2. Data preparation:

We use the MSMARCO dataset provided by [RIPOR](https://github.com/HansiZeng/RIPOR).

3. Run the experiments

For LLaMA:
```bash
# train
bash scripts/finetune_llama.sh
# eval loss
bash scripts/eval_loss_llama.sh
```

For T5:
```bash
# train
bash scripts/train_t5.sh
# eval loss
bash scripts/eval_loss_t5.sh
```


### Note
1. For both two methods, you can change the model name to test different sizes of models.
2. After evaluating the loss, you can calculate the contrastive generation loss according to the paper.
3. For inference scaling, you can change the beam size in MINDER test scripts to record the performance.

## Citation

If you use source code or dataset in your research, please cite our paper:
```bibtex

```

## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

## Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).
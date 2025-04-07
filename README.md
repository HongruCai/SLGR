
# Exploring Training and Inference Scaling Laws in Generative Retrieval

## 🔍 Overview

We study how model size, training data, and inference-time compute affect the performance of generative retrieval, a paradigm where LLMs generate document identifiers. To enable robust comparison, we introduce a new evaluation metric based on contrastive entropy and generation loss. Our results show that larger LLMs, especially decoder-only models like LLaMA, benefit more from increased inference compute. N-gram-based decoding aligns well with scaling trends, highlighting key design choices for future generative retrieval systems.

> For more details, refer to our paper accepted to **SIGIR 2025**: [Exploring Training and Inference Scaling Laws in Generative Retrieval](https://arxiv.org/abs/2503.18941).

## 📦 Requirements

To run the experiments, two different environments are required: one for MINDER_LLaMA and RIPOR, and another for MINDER_T5.

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

## 🧾 Data

We use the following datasets:

- MINDER experiments: NQ (Natural Questions) dataset.
- RIPOR experiments: MSMARCO dataset.
The preprocessed data and FMIndex are available for download on [Huggingface](https://huggingface.co/datasets/HenryCai/SLGR_data). Place the data in the `data` folder.

Although the FMIndex should work if the environment is set up correctly, we recommend rebuilding the FMIndex in your environment for best results.

## 📈 Experiments

### MINDER

[MINDER](https://arxiv.org/abs/2305.16675) is a generative retrieval method that uses text spans (e.g., body text, title, and pseudo-query) as document identifiers. For simplicity, we use only the body text as the document identifier.

#### MINDER_LLaMA
1. Install FMIndex:

Follow the instructions in the [SEAL](https://github.com/facebookresearch/SEAL) repository to install the necessary dependencies (you may need to clone the SEAL repo to install sdsl-lite).

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

The steps are similar to MINDER_LLaMA, but you will use a different environment.

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

We use the MSMARCO dataset provided by [RIPOR](https://github.com/HansiZeng/RIPOR) repository.

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
1. **Model Sizes**: For both methods, you can test different model sizes by changing the model name.
2. **CGL Calculation**: After evaluating the loss, you can calculate the contrastive generation loss as described in the paper.
3. **Inference Scaling**: For inference scaling, you can adjust the beam size in the MINDER test scripts to observe performance changes.

## 📚 Citation

If you use source code or dataset in your research, please cite our paper:
```bibtex
@inproceedings{cai2025exploringtraininginferencescaling,
  title={Large Language Models Empowered Personalized Web Agents},
  author={Hongru Cai and Yongqi Li and Ruifeng Yuan and Wenjie Wang and Zhen Zhang and Wenjie Li and Tat-Seng Chua},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series={SIGIR'25},
  year={2025}
}

```

## 📄 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

## 📬 Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).

# SLGR 

This repository contains the code and resources for three distinct experimental groups conducted in the study of Scaling Laws for Generative Retrieval. Each experiment is organized into its respective directory:

1. **Experiment Group 1**: MINDER_LLaMA, LlaMA models training and inference in MINDER
2. **Experiment Group 2**: MINDER_T5, T5 models training and inference in MINDER
3. **Experiment Group 3**: RIPOR, LLaMA and T5 models training and inference in RIPOR

## Getting Started

To replicate the experiments, follow these general steps:

1. **Clone the Repository**:
   ```bash
   git clone https://anonymous.4open.science/r/SLGR-16DB.git
   cd SLGR
   ```

2. **For experiments in MINDER_LlaMA**:
   ```bash
   cd MINDER_LLaMA 
   conda env create -f environment.yaml
   conda activate mllama
   # train
   bash scripts/finetune_llama.sh
   # test
   bash scripts/test_llama.sh
   # eval metric
   bash scripts/eval_loss.sh
   ```

3. **For experiments in MINDER_T5**:
   ```bash
   cd MINDER_T5
   conda env create -f environment.yaml
   conda activate mt5
   # train
   conda activate mllama
   bash scripts/train.sh
   # test
   conda activate mt5
   bash scripts/test_t5.sh
   # eval metric
   bash scripts/eval_loss.sh
   ```

4. **For experiments in RIPOR**:
   ```bash
   cd RIPOR
   conda activate mllama
   # train
   bash scripts/finetune_llama.sh
   bash scripts/train_t5.sh
   # eval metric
   bash scripts/eval_loss_llama.sh
   bash scripts/eval_loss_t5.sh
   ```


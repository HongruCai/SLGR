
# SLGR 

This repository contains the code and resources for three distinct experimental groups conducted in the study of Scaling Laws for Generative Retrieval (SLGR). Each experiment is organized into its respective directory:

1. **Experiment Group 1**: MINDER_LLaMA, LlaMA models training and inference in MINDER
2. **Experiment Group 2**: MINDER_T5, T5 models training and inference in MINDER
3. **Experiment Group 3**: RIPOR, LLaMA and T5 models training and inference in RIPOR

## Getting Started

To replicate the experiments, follow these general steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HongruCai/SLGR.git
   cd SLGR
   ```

2. **Navigate to the Desired Experiment Directory**:
   ```bash
   cd MINDER_LLaMA # or MINDER_T5, RIPOR
   ```

3. **Install Dependencies**:
   conda env create -f environment.yaml

4. **Run the Experiment**:
   bash scripts/train.sh
   bash scripts/test.sh


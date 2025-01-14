## 下载数据
```bash
cd SLGR
pip install gdown
gdown https://drive.google.com/drive/folders/1afwIZ3HYdj5YLDdG9ClBUnJUxgfFkPtg -O /data --folder
```

## 新建环境

```bash
cd MINDER_Llama
conda env create -f environment.yml
conda activate mllama
conda install -c conda-forge swig
conda install -c conda-forge cmake
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -e .
```

## 运行
需要确保有llama-2下载权限
```bash
cd MINDER_Llama #保持在MINDER_Llama目录下
bash scripts/full_pipeline.sh
bash scripts/eval_loss.sh #记录loss

#补充
# 不同数据集大小的实验
bash scripts/finetune_llama_frac.sh #会分别训练四个模型，使用0.2，0.4，0.6，0.8的数据集
bash scripts/eval_loss_frac.sh #记录loss，单卡
``` 



## 下载数据

```bash
cd SLGR
pip install gdown
gdown https://drive.google.com/drive/folders/1afwIZ3HYdj5YLDdG9ClBUnJUxgfFkPtg -O /data --folder
```

## 新建环境
```bash
cd MINDER_T5
conda env create -f environment.yml
conda activate mt5
conda install -c conda-forge swig
conda install -c conda-forge cmake
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
pip install -e .
```

## 运行

```bash
cd MINDER_T5 #保持在MINDER_T5目录下
conda activate mllama #训练需要使用MINDER_Llama构建的环境，才能使用peft
bash scripts/train.sh
conda activate mt5 #推理需要使用MINDER_T5构建的环境, 因为有些代码依赖于旧版本的transformers
bash scripts/record.sh #记录loss
bash scripts/eval.sh #记录recall
``` 



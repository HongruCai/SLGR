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
``` 



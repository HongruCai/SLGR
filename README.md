## 下载数据
从google drive下载数据集，将数据集放在`./data`文件夹下
```bash
cd SLGR
pip install gdown
gdown https://drive.google.com/drive/folders/1afwIZ3HYdj5YLDdG9ClBUnJUxgfFkPtg -O /data --folder
```

## 环境与运行
MINDER_Llama和MINDER_T5是两个不同的代码，需要两个不同的环境

先进入MINDER_Llama目录，根据其中的README新建环境并训练

再进入MINDER_T5目录，根据其中的README新建环境并训练
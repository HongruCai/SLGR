## 运行
这里与之前一样，需要训练5个不同大小的T5和3个不同大下的Llama
```bash
cd RIPOR
conda activate mllama #训练需要使用MINDER_Llama构建的环境
bash scripts/train_t5.sh #训练T5
bash scripts/record_loss_t5.sh #记录loss

bash scripts/finetune_llama.sh #训练Llama
bash scripts/record_loss_llama.sh #记录loss
``` 
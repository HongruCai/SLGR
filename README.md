
## 要跑的实验
1. MINDER_Llama （使用mllama环境）
- 训练使用full_pipeline.sh和finetune_llama_frac.sh （多卡）
- eval performance的代码还在修改，暂时不跑
- eval loss使用eval_loss.sh和eval_loss_frac.sh （单卡）

| 模型       | 训练 | Eval Performance (暂时不跑)       | Eval Loss |
|------------|----------|-------------|-------|
| Llama-7b   | 1   |          |     |
| Llama-13b  | 1  |         |     |
| Llama-70b      | 1   |         |     |
| Llama-7b_0.2  |    |          |     |
| Llama-7b_0.4  |    |          |     |
| Llama-7b_0.6  |    |          |     |
| Llama-7b_0.8  |    |          |     |

2. MINDER_T5 （训练使用mllama环境，eval使用mt5环境）
- 训练使用train.sh （多卡）
- eval performance使用eval.sh （单卡）
- eval loss使用eval_loss.sh （单卡）

| 模型       | 训练 | Eval Performance       | Eval Loss |
|------------|----------|-------------|-------|
| t5-small  | 1   |          |     |
| t5-base  | 1  |         |     |
| t5-large      | 1   |         |     |
| t5-3b  |   1 |          |     |
| t5-11b  |  1  |          |     |

3. RIPOR （使用mllama环境）
- 训练使用train_t5.sh，finetune_llama.sh，finetune_llama_frac.sh （多卡）
- eval loss使用record_loss_t5.sh, record_loss_llama.sh, record_loss_llama_frac.sh （多卡）

| 模型       | 训练 | Eval Loss |
|------------|----------|-------|
| t5-small  |    |     |
| t5-base  |   |     |
| t5-large   |    |     |
| t5-3b  |    |     |
| t5-11b  |    |     |
| llama-7b  |    |     |
| llama-13b  |   |     |
| llama-70b  |    |     |
| llama-7b_0.2  |   |     |
| llama-7b_0.4  |    |     |
| llama-7b_0.6  |    |     |
| llama-7b_0.8  |    |     |



## 最后的eval performance

### MINDER_Llama文件夹

```bash
conda activate mllama
bash scripts/test_llama.sh
```
单卡，可以修改jobs, 但是注意batch size只能为1，否则会有bug

切换模型需要修改backbone, checkpoint, output

Llama-70b如果无法单卡运行，则放弃，因为使用decive_map='auto'会报错

### MIDNER_T5文件夹

```bash
conda activate mt5
bash scripts/test_t5.sh
```
单卡，可以修改jobs, batch size也可以增大
切换模型需要修改backbone, checkpoint, output

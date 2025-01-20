
## 最后的eval performance

### MINDER_Llama文件夹

```bash
bash scripts/test_llama.sh
```
单卡，可以修改jobs, 但是注意batch size只能为1，否则会有bug

切换模型需要修改backbone, checkpoint, output

llama-70b可能无法运行，不跑

### MIDNER_T5文件夹

```bash
bash scripts/test_t5.sh
```
单卡，可以修改jobs, batch size也可以增大
切换模型需要修改backbone, checkpoint, output

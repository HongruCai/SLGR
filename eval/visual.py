import numpy as np
import matplotlib.pyplot as plt

# 原始数据
N = np.array([44057088, 198229248, 704769024, 2818699264, 11274422272], dtype=np.float32)  
loss = np.array([0.003739327519766013, 0.00365354820880983,  0.003623691318817766, 0.0035889147738936734,0.003575122915571207], dtype=np.float32)  

# 拟合参数
a_opt = 0.00585561  # a
b_opt = 0.37771821    # b
c_opt = 0.00355248   # c

# 线性化数据
log_N = np.log(N)
log_loss = np.log(loss - c_opt)

# 绘制数据点
plt.scatter(log_N, log_loss, label='Data Points', color='blue')

# 绘制拟合直线
log_fit = np.log(a_opt) - b_opt * log_N
plt.plot(log_N, log_fit, linestyle='--', label='Fitted Curve', color='black')

# 美化图表
plt.xlabel("Log(Number of Non-Embedding Parameters)")
plt.ylabel("Log(Contrastive Entropy - c)")
plt.title("Scaling Laws for Loss")
plt.legend()
plt.grid()

plt.savefig('./result/scaling_law_vis.png')
plt.close()
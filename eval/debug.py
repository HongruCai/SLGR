import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义原始函数
def loss_function(n, a, b, c):
    return (a / n)**b + c

# 生成模拟数据
n_data = np.linspace(1, 100, 50)  # n 的范围
true_a, true_b, true_c = 10, 2, 5  # 真值
loss_data = loss_function(n_data, true_a, true_b, true_c) + np.random.normal(0, 0.5, size=len(n_data))  # 加噪声

# 对数变换后的拟合函数
def linearized_loss(log_n, log_a, b, c):
    return b * (-log_n) + log_a + np.log(c)

# 直接拟合非线性模型
popt, pcov = curve_fit(loss_function, n_data, loss_data, p0=[1, 1, 1])  # 初值 [a, b, c]
fitted_a, fitted_b, fitted_c = popt

# 打印拟合结果
print(f"拟合参数: a = {fitted_a:.3f}, b = {fitted_b:.3f}, c = {fitted_c:.3f}")

# 可视化拟合结果
plt.figure(figsize=(10, 5))

# 原始数据
plt.scatter(n_data, loss_data, label="Data (with noise)", color="blue")
# plt.plot(n_data, loss_function(n_data, true_a, true_b, true_c), label="True Function", color="green", linestyle="--")

# 拟合曲线
plt.plot(n_data, loss_function(n_data, fitted_a, fitted_b, fitted_c), label="Fitted Function", color="red")

# 图形设置
plt.xlabel("n")
plt.ylabel("loss")
plt.title("Nonlinear Fit of Loss Function")
plt.legend()
plt.savefig('./result/debu.png', dpi=300, bbox_inches='tight')
plt.show()



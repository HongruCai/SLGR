import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# 定义拟合函数
def scaling_law(params, n):
    a, b, c = params
    return (a / n) ** b + c

# 定义残差函数
def residuals(params, n, l):
    return scaling_law(params, n) - l

# 定义函数用于拟合数据并计算 R²
def fit_scaling_law(n, l, restarts=50):
    best_result = None
    best_r_squared = -float('inf')  # 初始化为负无穷

    for _ in range(restarts):  # 重启次数
        # 随机生成初始值
        initial_guess = np.random.uniform(low=[1, 0.1, -1], high=[1e4, 2, 1], size=3)
        result = least_squares(residuals, x0=initial_guess, args=(n, l))
        
        # 获取拟合结果
        predicted_l = scaling_law(result.x, n)
        
        # 计算 R²
        ss_res = np.sum((l - predicted_l) ** 2)
        ss_tot = np.sum((l - np.mean(l)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 更新最优结果
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_result = result

    return best_result, best_r_squared

# 数据组 1
n1 = np.array([44057088, 198229248, 704769024, 2818699264, ], dtype=np.float32)
l1 = np.array([46.6, 50.3, 53.5, 54.6, ], dtype=np.float32)

# 数据组 2
# n2 = np.array([6607343616, 12852024320, 68714504192], dtype=np.float32)
# l2 = np.array([0.003349998694460214, 0.003293957555899526, 0.0032800482437811704], dtype=np.float32)

# 拟合数据组 1
result1, r_squared1 = fit_scaling_law(n1, l1)
a1, b1, c1 = result1.x
n1_smooth = np.logspace(np.log10(n1.min()), np.log10(n1.max()), 1000)
predicted_l1 = scaling_law(result1.x, n1_smooth)

n_new = 11274422272 # 例如，模型大小为 5000

# 使用最优参数计算预测值
l_new = scaling_law(result1.x, np.array([n_new]))

print(f"对于模型大小 n={n_new}，预测的 Loss 值为 {l_new[0]}")

# 拟合数据组 2
# result2, r_squared2 = fit_scaling_law(n2, l2)
# a2, b2, c2 = result2.x
# n2_smooth = np.logspace(np.log10(n2.min()), np.log10(n2.max()), 1000)
# predicted_l2 = scaling_law(result2.x, n2_smooth)

# 绘制图像
plt.figure(figsize=(10, 7))
plt.scatter(n1, l1, color='red', label='Data Group 1')
plt.plot(n1_smooth, predicted_l1, color='blue', label=f'Fit Group 1\n$a={a1:.4f}$, $b={b1:.4f}$, $c={c1:.4f}$\n$R^2={r_squared1:.4f}$')

# plt.scatter(n2, l2, color='orange', label='Data Group 2')
# plt.plot(n2_smooth, predicted_l2, color='green', label=f'Fit Group 2\n$a={a2:.4f}$, $b={b2:.4f}$, $c={c2:.4f}$\n$R^2={r_squared2:.4f}$')

plt.xscale('log')  # 对数刻度
plt.xlabel("Model Size (n)")
plt.ylabel("Metric (l)")
plt.title("Scaling Law Fitting for Two Data Groups")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('./result/scaling_law_two_groups.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印结果
print(f"Group 1: Scaling Law: l = ({a1:.4f} / n) ^ {b1:.4f} + {c1:.4f}, R² = {r_squared1:.4f}")
# print(f"Group 2: Scaling Law: l = ({a2:.4f} / n) ^ {b2:.4f} + {c2:.4f}, R² = {r_squared2:.4f}")

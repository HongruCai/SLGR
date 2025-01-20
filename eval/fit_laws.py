import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
# 输入数据
# n = np.array([60000000, 220000000, 770000000, 3000000000, 7000000000], dtype=np.float32)  
# l = np.array([5.111778953, 5.074381816, 5.066202679, 5.032914397, 4.970532825], dtype=np.float32)  

# MINDER T5
# n = np.array([44057088, 198229248, 704769024, 2818699264, 11274422272], dtype=np.float32)  
# l = np.array([0.003732272730265805, 0.003646773468531139, 0.0036170537611817516, 0.003582365477361566,0.003568628382401702], dtype=np.float32)  

# MINER LLAMA
# n = np.array([6607343616, 12852024320, 68714504192], dtype=np.float32)
# l = np.array([0.0033441422984040446, 0.0032882237133081, 0.0032743631248782932], dtype=np.float32)

# MINDER LLAMA Data
# n = np.array([117760, 235520, 353280, 471040, 588800], dtype=np.float32)
n = np.array([ 235520, 353280,  588800], dtype=np.float32)

l = np.array([ 5.29854055059404, 5.293349691955132,  5.252856668907272], dtype=np.float32)


# 定义拟合函数
def scaling_law(params, n):
    a, b, c = params
    return (a / n) ** b + c

# 定义残差函数
def residuals(params, n, l):
    return scaling_law(params, n) - l

# 执行最小二乘法拟合
# initial_guess = [100, 1, 0]
# result = least_squares(residuals, x0=initial_guess, args=(n, l))

# 全局优化拟合
best_result = None
best_r_squared = -float('inf')  # 初始化为负无穷，因为 R^2 越大越好

for _ in range(50):  # 重启次数
    # 随机生成初始值
    initial_guess = np.random.uniform(low=[1, 0.1, -1], high=[1000, 2, 1], size=3)
    result = least_squares(residuals, x0=initial_guess, args=(n, l))
    
    # 获取拟合结果
    predicted_l = scaling_law(result.x, n)
    
    # 计算 R^2
    ss_res = np.sum((l - predicted_l) ** 2)
    ss_tot = np.sum((l - np.mean(l)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 更新最优结果
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_result = result

# 获取最优参数
a_opt, b_opt, c_opt = best_result.x

# 使用最优参数计算预测值
predicted_l = scaling_law(best_result.x, n)

# 平滑曲线
n_smooth = np.logspace(np.log10(n.min()), np.log10(n.max()), 1000)
predicted_l_smooth = scaling_law(best_result.x, n_smooth)


# 绘制拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(n, l, color='red', label='Data Points')
plt.plot(n_smooth, predicted_l_smooth, label=f'Fitted Curve\n$a={a_opt:.4f}$, $b={b_opt:.4f}$, $c={c_opt:.4f}$\n$R^2={r_squared:.4f}$', color='blue')
# plt.xscale('log')  # 使用对数刻度展示更直观
# plt.yscale('log')
plt.xlabel("Model Size")
plt.ylabel("Metric")
plt.title("Scaling Law Fitting (Least Squares - SciPy)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('./result/scaling_law_fit_scipy.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印结果
print(f"Scaling Law: l = ({a_opt:.4f} / n) ^ {b_opt:.4f} + {c_opt:.4f}")
print(f"R²: {r_squared:.4f}")
print(f"Optimal Parameters: {result.x}")
print(f"Optimization Status: {result.message}")



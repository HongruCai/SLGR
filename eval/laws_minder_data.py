import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

# MINDER LLAMA Data
# n = np.array([117760, 235520, 353280, 471040, 588800], dtype=np.float32)
n = np.array([ 117760,  235520, 353280,471040, 588800], dtype=np.float32)
l = np.array([0.003413291978135587, 0.003353927870270544,0.0033509868940814346,0.0033502276147056305, 0.003349998694460214], dtype=np.float32)


# RIPOR T5
# n = np.array([44057088, 198229248, 704769024, 2818699264, 11274422272], dtype=np.float32)
# l = np.array([0.003889985249666699, 0.0038899180041356693,  0.003893088394366349, 0.003888390087815426,0.003889115035925551], dtype=np.float32)


# MINDER T5
# n = np.array([44057088, 198229248, 704769024, 2818699264, 11274422272], dtype=np.float32)  
# l = np.array([0.003739327519766013, 0.00365354820880983,  0.003618691318817766, 0.0035889147738936734,0.003575122915571207], dtype=np.float32)  

# MINER LLAMA
# n = np.array([6607343616, 12852024320, 68714504192], dtype=np.float32)
# l = np.array([0.003349998694460214, 0.003293957555899526, 0.0032800482437811704], dtype=np.float32)



# 定义拟合函数
def scaling_law(params, n):
    a, b, c = params
    return (a / n) ** b + c

# 定义残差函数
def residuals(params, n, l):
    return scaling_law(params, n) - l


# 全局优化拟合
best_result = None
best_r_squared = -float('inf')  

for _ in range(50):  # 重启次数
    # 随机生成初始值
    initial_guess = np.random.uniform(low=[1, 0.1, -1], high=[1e4, 2, 1], size=3)
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
print(f"Optimal Parameters: {best_result.x}")
# 使用最优参数计算预测值
predicted_l = scaling_law(best_result.x, n)

# n_new = 471040 # 例如，模型大小为 5000

# # 使用最优参数计算预测值
# l_new = scaling_law(best_result.x, np.array([n_new]))

# print(f"对于模型大小 n={n_new}，预测的 Loss 值为 {l_new[0]}")

# 平滑曲线
# n_smooth = np.logspace(np.log10(n.min()), np.log10(n.max()), 10000)
# predicted_l_smooth = scaling_law(best_result.x, n_smooth)


# plt.figure(figsize=(8, 6))
# plt.scatter(n, l, color='red', label='Data Points')
# plt.plot(n_smooth, predicted_l_smooth, label=f'Fitted Curve\n$a={a_opt}$, $b={b_opt}$, $c={c_opt}$\n$R^2={r_squared:.4f}$', color='blue')
# # plt.xscale('log')  # 使用对数刻度展示更直观
# # plt.yscale('log')
# plt.xlabel("Model Size")
# plt.ylabel("Metric")
# plt.title("Scaling Law Fitting (Least Squares - SciPy)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig('./result/scaling_law_fit.png', dpi=300, bbox_inches='tight')
# plt.close()

# 打印结果
print(f"Scaling Law: l = ({a_opt} / n) ^ {b_opt} + {c_opt}")
print(f"R²: {r_squared:.4f}")
print(f"Optimal Parameters: {result.x}")

adjusted_l = l - c_opt  
print(adjusted_l)
adjusted_l = np.where(adjusted_l > 0, adjusted_l, predicted_l - c_opt)

def log_to_original(y, _):
    """将 log 值还原为原始值的格式化函数"""
    return f"{(np.exp(y)+c_opt):.6f}"


log_n = np.log(n)
log_l = np.log(adjusted_l)

# 平滑曲线（为 log-log 形式）
n_smooth = np.logspace(np.log10(n.min()), np.log10(n.max()), 10000)
predicted_l_smooth = scaling_law(best_result.x, n_smooth)
log_predicted_l_smooth = np.log(predicted_l_smooth - c_opt)


plt.figure(figsize=(6, 2.5))  # 宽高可以根据需要调整，3.5x3.5 英寸适合一栏
plt.rcParams['font.size'] = 12  # 设置字体大小为 14pt
plt.rcParams['font.weight'] = 'bold'
# 绘制散点图
plt.scatter(n, log_l, color='#4e87b2', s=100, label='LLaMA-7B')

# 绘制拟合曲线
plt.plot(n_smooth, log_predicted_l_smooth, color='#4d4d4d', linestyle='--', linewidth=3)

# 设置对数横轴
plt.xscale('log')

# 自定义纵轴标签格式（原始值）
plt.gca().yaxis.set_major_formatter(FuncFormatter(log_to_original))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5)) 
plt.legend()
# plt.ylim(log_l.min() - 0.1, log_l.max() + 0.1)  # 稍微扩展范围

# 保存图片
plt.savefig('./result/scaling_law_minder_data.png', dpi=300, bbox_inches='tight')
plt.savefig('./result/scaling_law_minder_data.svg', format="svg", bbox_inches='tight')
plt.show()



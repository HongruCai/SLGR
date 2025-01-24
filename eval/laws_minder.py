import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.ticker import MaxNLocator

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
n1 = np.array([44057088, 198229248, 704769024, 2818699264, 11274422272], dtype=np.float32)
l1 = np.array([0.003739327519766013, 0.00365354820880983, 0.003618691318817766, 0.0035889147738936734, 0.003575122915571207], dtype=np.float32)

# 数据组 2
n2 = np.array([6607343616, 12852024320, 68714504192], dtype=np.float32)
l2 = np.array([0.003349998694460214, 0.003293957555899526, 0.0032800482437811704], dtype=np.float32)

# 拟合数据组 1
result1, r_squared1 = fit_scaling_law(n1, l1)
a1, b1, c1 = result1.x
n1_smooth = np.logspace(np.log10(n1.min()), np.log10(n1.max()), 1000)
predicted_l1 = scaling_law(result1.x, n1_smooth)

# 拟合数据组 2
result2, r_squared2 = fit_scaling_law(n2, l2)
a2, b2, c2 = result2.x
n2_smooth = np.logspace(np.log10(n2.min()), np.log10(n2.max()), 1000)
predicted_l2 = scaling_law(result2.x, n2_smooth)



# 绘制图像
plt.figure(figsize=(6, 2.5))
# plt.rcParams['font.family'] = 'Calibri'  # 设置字体为 Arial
plt.rcParams['font.size'] = 12  # 设置字体大小为 14pt
plt.rcParams['font.weight'] = 'bold'
plt.scatter(n1, l1, color='#f3d266', label='T5 Series', s=100)
plt.plot(n1_smooth, predicted_l1, color='#4d4d4d', linestyle='--',linewidth=3)

plt.scatter(n2, l2, color='#4e87b2', label='LLaMA Series',s=100)
plt.plot(n2_smooth, predicted_l2, color='#4d4d4d', linestyle='--',linewidth=3)

plt.axhline(c1, color='#f3d266', linestyle=':',linewidth=3)
plt.axhline(c2, color='#4e87b2', linestyle=':',linewidth=3)


plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))  
# 实验部分的图，intro的图则注释掉
plt.ylim(0.0031, 0.00386)
plt.xscale('log')
plt.xlim(0, 1e11)

plt.legend(loc='upper right')
plt.savefig('./result/scaling_law_minder_intro.png', bbox_inches='tight')
plt.savefig('./result/scaling_law_minder_intro.svg', format="svg", bbox_inches='tight')
plt.show()

# 打印结果
print(f"Group 1: Scaling Law: l = ({a1} / n) ^ {b1} + {c1}, R² = {r_squared1 }")
print(f"Group 2: Scaling Law: l = ({a2} / n) ^ {b2} + {c2 }, R² = {r_squared2 }")

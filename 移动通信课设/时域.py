import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# --------------仿真场景设定----------------
L = np.arange(1, 1001)          # 时间索引 1 到 1000
N = 16                          # 散射波数量
v = 40                          # 移动速度 40 km/h
f_c = 800e6                     # 载波频率 800 MHz
C = 3e8                         # 光速 3×10^8 m/s
R = 9600                        # 码速率 9600 bps
_lambda = C / f_c               # 波长
t_s = 1 / R                     # 采样时间间隔

# --------------多径信号到达角与相位初始化----------------
fDT = (v * 1000/3600) / C * f_c  # 注意：v 从 km/h 转为 m/s
theta = 2 * np.pi * np.random.rand(N)
phi   = 2 * np.pi * np.random.rand(N)

# 预分配阵列
fading_I   = np.zeros_like(L, dtype=float)
fading_Q   = np.zeros_like(L, dtype=float)
fading_log = np.zeros_like(L, dtype=float)

# --------------循环计算瑞利衰落----------------
for k in range(len(L)):
    T_I = 0.0
    T_Q = 0.0
    # 多径叠加
    for n in range(N):
        arg = 2 * np.pi * fDT * (k+1) * t_s * np.cos(theta[n]) + phi[n]
        T_I += np.cos(arg)
        T_Q += np.sin(arg)
    # 归一化
    fading_I[k] = T_I / np.sqrt(N)
    fading_Q[k] = T_Q / np.sqrt(N)

    # 计算幅度并转换为 dB
    fading_abs   = np.sqrt(fading_I[k]**2 + fading_Q[k]**2)
    fading_log[k] = 20 * np.log10(fading_abs)

# --------------绘图显示----------------
plt.figure(figsize=(8,4))
plt.plot(L, fading_log)
plt.title('瑞利衰落信道时域变化')
plt.xlabel('时间样本 k')
plt.ylabel('信道衰落 (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()

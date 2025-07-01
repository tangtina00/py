import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# --------------仿真场景设定----------------
N = 16               # 散射波数量
x_max = 100          # x 轴最大值 (cm)
y_max = 100          # y 轴最大值 (cm)
f_c = 800e6          # 载波频率 800 MHz
C = 3e8              # 光速 3×10^8 m/s
ramda = C / f_c      # 波长 (m)
NORM = 1 / np.sqrt(N)

# 预分配
fading_I   = np.zeros((x_max, y_max))
fading_Q   = np.zeros((x_max, y_max))
fading_abs = np.zeros((x_max, y_max))

# --------------多径信号到达角与相位初始化----------------
# θ 从一个随机初始相位开始，每步递增 2π/16
offset = np.random.rand() * 2 * np.pi / 16
theta  = offset + np.arange(N) * (2 * np.pi / 16)
# φ 为 [0,2π) 上均匀分布
phi    = 2 * np.pi * np.random.rand(N)

# --------------空间网格循环计算----------------
for ix in range(x_max):
    for iy in range(y_max):
        # 将位置 cm 转为 m: ix/100, iy/100
        x_m = ix / 100.0
        y_m = iy / 100.0
        
        # 计算每条多径的相位偏移 psi
        psi = ((y_m * np.cos(theta) - x_m * np.sin(theta))
               * 2 * np.pi / ramda)
        
        # I/Q 分量叠加
        F_I = NORM * np.sum(np.cos(psi + phi))
        F_Q = NORM * np.sum(np.sin(psi + phi))
        
        fading_I[ix, iy]   = F_I
        fading_Q[ix, iy]   = F_Q
        fading_abs[ix, iy] = np.sqrt(F_I**2 + F_Q**2)

# --------------三维绘图显示----------------
X = np.arange(1, x_max + 1)
Y = np.arange(1, y_max + 1)
X, Y = np.meshgrid(X, Y, indexing='ij')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    X, Y, fading_abs,
    rstride=4, cstride=4, cmap='viridis', edgecolor='none'
)
ax.set_title('瑞利衰落信道空域变化')
ax.set_xlabel('X 位置 (cm)')
ax.set_ylabel('Y 位置 (cm)')
ax.set_zlabel('衰落幅度')
plt.tight_layout()
plt.show()

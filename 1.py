import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置时间变量 t
t = np.linspace(0, 2 * np.pi, 300)
cos_t = np.cos(t)
sin_t = np.sin(t)
exp_jt = np.exp(1j * t)

# 创建图形和子图
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("单位圆上的复数旋转：$e^{j\\theta}$")

# 初始化元素
point, = ax.plot([], [], 'ro', label=r'$e^{j\theta}$')
cos_line, = ax.plot([], [], 'b--', label='Re = cos(θ)')
sin_line, = ax.plot([], [], 'g--', label='Im = sin(θ)')
radius_line, = ax.plot([], [], 'r-', lw=1)

ax.legend(loc='upper right')

# 动画初始化函数
def init():
    point.set_data([], [])
    cos_line.set_data([], [])
    sin_line.set_data([], [])
    radius_line.set_data([], [])
    return point, cos_line, sin_line, radius_line

# 更新帧函数
def update(frame):
    z = exp_jt[frame]
    x, y = np.real(z), np.imag(z)
    point.set_data([x], [y])
    cos_line.set_data([0, x], [0, 0])
    sin_line.set_data([x, x], [0, y])
    radius_line.set_data([0, x], [0, y])
    return point, cos_line, sin_line, radius_line

# 创建动画
ani = animation.FuncAnimation(
    fig, update, frames=len(t),
    init_func=init, blit=True, interval=30
)

plt.close(fig)
ani

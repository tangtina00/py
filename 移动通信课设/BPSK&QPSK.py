import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 参数设置
M = 4                # 符号数，减少以便观察
fc = 20            # 载波频率 (Hz)
Rs = 2               # 符号率 (symbols/s)，这样每符号持续 0.5s
Fs = 10000           # 采样率 (Hz)
Ts = 1 / Rs          # 符号时长 (s)
samples_per_symbol = int(Fs / Rs)

# 生成随机比特
a_bpsk = (np.random.rand(M) > 0.5).astype(int)
I_bpsk = 2 * a_bpsk - 1

# 时间轴仅覆盖前 2 个符号
t = np.arange(0, 2 * Ts, 1 / Fs)
s_bpsk = np.zeros_like(t)

# 合成 BPSK 正弦波（使用 sine 形式）
for i in range(2):
    idx = slice(i * samples_per_symbol, (i + 1) * samples_per_symbol)
    s_bpsk[idx] = I_bpsk[i] * np.sin(2 * np.pi * fc * t[idx])

plt.figure(figsize=(8, 3))
plt.plot(t, s_bpsk, linewidth=1.5)
plt.title('BPSK 调制时域')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.xlim(0, 2 * Ts)
plt.grid(True)
plt.show()

# QPSK
# 确保偶数比特
if M % 2 != 0:
    M += 1
a_qpsk = (np.random.rand(M) > 0.5).astype(int)
I_qpsk = 2 * a_qpsk[0::2] - 1
Q_qpsk = 2 * a_qpsk[1::2] - 1
N_sym = len(I_qpsk)

# 时间轴仅前 2 符号
t2 = np.arange(0, 2 * Ts, 1 / Fs)
s_qpsk = np.zeros_like(t2)

for i in range(2):
    idx = slice(i * samples_per_symbol, (i + 1) * samples_per_symbol)
    s_qpsk[idx] = (I_qpsk[i] * np.sin(2 * np.pi * fc * t2[idx]) -
                   Q_qpsk[i] * np.cos(2 * np.pi * fc * t2[idx]))

plt.figure(figsize=(8, 3))
plt.plot(t2, s_qpsk, linewidth=1.5)
plt.title('QPSK 调制时域')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.xlim(0, 2 * Ts)
plt.grid(True)
plt.show()

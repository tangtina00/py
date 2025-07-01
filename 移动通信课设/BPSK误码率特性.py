import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 仿真参数
N_sim = int(1e6)                  # 符号数
EbN0_dB = np.arange(0, 21, 1)    # Eb/N0 范围 (dB)
N_path = 16                      # 多径数目
fDT = 0.01                       # 归一化多普勒频率

# 预生成衰落信道
k = np.arange(1, N_sim + 1)
theta1 = 2 * np.pi * np.random.rand()
theta_n = theta1 + 2 * np.pi * np.arange(N_path) / N_path
phi_n = 2 * np.pi * np.random.rand(N_path)
arg = 2 * np.pi * fDT * np.outer(np.cos(theta_n), k) + phi_n[:, None]
fading = (np.sum(np.cos(arg), axis=0) + 1j * np.sum(np.sin(arg), axis=0)) / np.sqrt(N_path)
fading_real = np.real(fading)
fading_imag = np.imag(fading)

# 初始化误码率数组
BER_bpsk = np.zeros_like(EbN0_dB, dtype=float)

# BPSK 调制解调
for idx, db in enumerate(EbN0_dB):
    EbN0 = 10 ** (db / 10)
    noise_std = 1 / np.sqrt(2 * EbN0)

    # 生成随机比特流
    bits = np.random.randint(0, 2, N_sim)
    symbols = 2 * bits - 1  # BPSK 映射为 +1 / -1

    # 生成高斯噪声
    noise = noise_std * np.random.randn(N_sim)

    # 通过 瑞利 衰落信道
    r = symbols * fading_real + noise

    # 同步检波（最大比合并）
    r_mrc = r * fading_real

    # 判决
    bits_hat = (r_mrc >= 0).astype(int)

    # 计算误码率
    BER_bpsk[idx] = np.mean(bits_hat != bits)

# QPSK 部分（保持不变）
EbN0_dB_qpsk = np.arange(0, 21, 1)
SER = np.zeros_like(EbN0_dB_qpsk, dtype=float)

# Gray 码 QPSK 映射函数
def qpsk_map(bits_i, bits_q):
    val = (1/np.sqrt(2)) * ((1 - 2*bits_i) + 1j*(1 - 2*bits_q))
    return np.real(val), np.imag(val)

# QPSK 仿真
for idx, db in enumerate(EbN0_dB_qpsk):
    EbN0 = 10 ** (db / 10)
    EsN0 = 2 * EbN0

    bits = np.random.randint(0, 2, size=(2, N_sim // 2))
    I_sym, Q_sym = qpsk_map(bits[0], bits[1])

    noise_std = np.sqrt(1 / (2 * EsN0))
    noise = noise_std * np.random.randn(N_sim // 2)

    r_I = I_sym * fading_real[:N_sim//2] - Q_sym * fading_imag[:N_sim//2] + noise
    r_Q = Q_sym * fading_real[:N_sim//2] + I_sym * fading_imag[:N_sim//2] + noise

    y_I = r_I * fading_real[:N_sim//2] + r_Q * fading_imag[:N_sim//2]
    y_Q = r_Q * fading_real[:N_sim//2] - r_I * fading_imag[:N_sim//2]

    bits_hat_i = (y_I < 0).astype(int)
    bits_hat_q = (y_Q < 0).astype(int)

    errors = np.sum((bits_hat_i != bits[0]) | (bits_hat_q != bits[1]))
    SER[idx] = errors / (N_sim/2)

# 理想 AWGN QPSK SER 曲线
EbN0_dB_ideal = np.arange(0, 13, 1)
SER_ideal = 2 * norm.cdf(-np.sqrt(2 * 10**(EbN0_dB_ideal/10))) * \
            (1 - 0.5 * norm.cdf(-np.sqrt(2 * 10**(EbN0_dB_ideal/10))))

# 绘图
plt.figure(figsize=(10, 6))
plt.semilogy(EbN0_dB, BER_bpsk, 'g-o', label='BPSK (瑞利)')
# plt.semilogy(EbN0_dB_qpsk, SER, 'r-o', label='QPSK (瑞利)')
plt.semilogy(EbN0_dB_ideal, SER_ideal, 'b-s', label='AWGN 理论')
plt.xlabel(r'$E_b/N_0\ (dB)$')
plt.ylabel('误码率 / 符号错误率')
plt.title('BPSK在瑞利衰落信道与AWGN的误码性能对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

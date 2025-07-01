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

# 随机生成衰落相位和到达角起始偏移
theta1 = 2 * np.pi * np.random.rand()
theta_n = theta1 + 2 * np.pi * np.arange(N_path) / N_path

# 预先生成 k 用于衰落系数
k = np.arange(1, N_sim//2 + 1)

# 计算衰落系数 Fk
phi_n = 2 * np.pi * np.random.rand(N_path)
arg = 2 * np.pi * fDT * np.outer(np.cos(theta_n), k) + phi_n[:, None]
fading = (np.sum(np.cos(arg), axis=0) + 1j * np.sum(np.sin(arg), axis=0)) / np.sqrt(N_path)
Fk_real = np.real(fading)
Fk_imag = np.imag(fading)

# 初始化结果
SER = np.zeros_like(EbN0_dB, dtype=float)

# Gray 码 QPSK 符号映射函数
def qpsk_map(bits_i, bits_q):
    # 输入 0/1, 输出对应 I/Q 值
    val = (1/np.sqrt(2)) * ((1 - 2*bits_i) + 1j*(1 - 2*bits_q))
    return np.real(val), np.imag(val)

# 仿真主循环
for idx, db in enumerate(EbN0_dB):
    EbN0 = 10 ** (db / 10)
    EsN0 = 2 * EbN0
    SNR = 2 * EsN0

    # 随机生成比特
    bits = np.random.randint(0, 2, size=(2, N_sim//2))  # 第一行为I位, 第二行为Q位
    
    # 调制
    I_sym, Q_sym = qpsk_map(bits[0], bits[1])
    
    # 生成噪声
    noise_var = (1/EsN0) / 2
    noise = np.sqrt(noise_var) * np.random.randn(N_sim//2)

    # 通过Rayleigh衰落信道 + AWGN
    r_I = I_sym * Fk_real - Q_sym * Fk_imag + noise
    r_Q = Q_sym * Fk_real + I_sym * Fk_imag + noise
    
    # 同步检波
    y_I = r_I * Fk_real + r_Q * Fk_imag
    y_Q = r_Q * Fk_real - r_I * Fk_imag

    # 判决
    bits_hat_i = (y_I < 0).astype(int)
    bits_hat_q = (y_Q < 0).astype(int)

    # 计算SER
    errors = np.sum((bits_hat_i != bits[0]) | (bits_hat_q != bits[1]))
    SER[idx] = errors / (N_sim/2)

# 理想AWGN QPSK SER 曲线
EbN0_dB_ideal = np.arange(0, 13, 1)
SER_ideal = 2 * norm.cdf(-np.sqrt(2 * 10**(EbN0_dB_ideal/10))) * \
            (1 - 0.5 * norm.cdf(-np.sqrt(2 * 10**(EbN0_dB_ideal/10))))

# 绘图
plt.semilogy(EbN0_dB, SER, 'r-o', label='Rayleigh SER')
plt.semilogy(EbN0_dB_ideal, SER_ideal, 'b-s', label='Ideal AWGN SER')
plt.xlabel(r'$E_b/N_0\ (dB)$')
plt.ylabel('符号错误率 (SER)')
plt.title('QPSK 在瑞利衰落与理想 AWGN 下的 SER 性能')
plt.legend()
plt.grid(True)
plt.show()

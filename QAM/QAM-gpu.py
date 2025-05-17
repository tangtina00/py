import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# 设置中文字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------------------------
# 1. 参数配置：确保使用GPU（NVIDIA 4060 Laptop, CUDA 12）
# ---------------------------------------------
# 检查 CUDA 版本和可用性
print(f"PyTorch CUDA 支持: {torch.cuda.is_available()}, CUDA 版本: {torch.version.cuda}")
# 强制设置 GPU 设备，假设只有一块GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# QAM阶数列表
qam_orders = [2, 4, 16, 64]
# SNR范围（dB）
snr_db = np.arange(0, 21, 2)
# 每个SNR点符号数量
num_symbols = 100000
# 训练参数
learning_rate = 0.001
batch_size = 1024
num_epochs = 10

# ---------------------------------------------
# 2. QAM调制函数
# ---------------------------------------------
def qam_modulate(bits, M):
    """
    对比特序列进行M-QAM调制
    bits: 0/1数组，长度为log2(M)整数倍
    M: 星座点数
    返回: 复数基带符号
    """
    k = int(np.log2(M))
    bit_groups = bits.reshape(-1, k)
    symbols = bit_groups.dot(1 << np.arange(k)[::-1])
    # 构造I/Q映射
    m_sqrt = int(np.sqrt(M))
    I = 2 * (symbols % m_sqrt) - m_sqrt + 1
    Q = 2 * (symbols // m_sqrt) - m_sqrt + 1
    return I + 1j * Q

# ---------------------------------------------
# 3. 理想解调函数（最小欧氏距离）
# ---------------------------------------------
def qam_demodulate(rx_symbols, M):
    re = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
    im = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
    constellation = np.array([x + 1j*y for x in re for y in im])
    idx = np.argmin(abs(rx_symbols.reshape(-1, 1) - constellation.reshape(1, -1)), axis=1)
    k = int(np.log2(M))
    bits = (((idx[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
    return bits

# ---------------------------------------------
# 4. 全连接神经网络解调模型
# ---------------------------------------------
class QAMNet(nn.Module):
    def __init__(self, M):
        super(QAMNet, self).__init__()
        k = int(np.log2(M))
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, M)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------
# 5. 主流程：训练并测试BER性能
# ---------------------------------------------
if __name__ == '__main__':
    results = {}
    for M in qam_orders:
        # 传统与DL BER 存储
        ber_traditional, ber_dl = [], []

        # 初始化并移动模型到GPU
        model = QAMNet(M).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 准备训练集（SNR=10dB）
        bits = np.random.randint(0, 2, int(num_symbols * np.log2(M)))
        symb = qam_modulate(bits, M)
        snr_train = 10**(10/10)
        noise = (np.random.randn(len(symb)) + 1j*np.random.randn(len(symb))) * np.sqrt(1/(2*snr_train))
        rx = symb + noise
        X = torch.from_numpy(np.vstack([rx.real, rx.imag]).T).float().to(device)
        y = torch.from_numpy(
            bits.reshape(-1, int(np.log2(M))).dot(1 << np.arange(int(np.log2(M)))[::-1])
        ).long().to(device)

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                             batch_size=batch_size, shuffle=True)

        # 训练
        for epoch in range(num_epochs):
            model.train()
            for xb, yb in loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 测试各SNR点
        for db in snr_db:
            snr_lin = 10**(db/10)
            bits_test = np.random.randint(0, 2, int(num_symbols * np.log2(M)))
            symb_test = qam_modulate(bits_test, M)
            noise = (np.random.randn(len(symb_test)) + 1j*np.random.randn(len(symb_test)))
            noise *= np.sqrt(1/(2*snr_lin))
            rx_test = symb_test + noise

            # 传统解调 BER
            bits_hat = qam_demodulate(rx_test, M)
            ber_traditional.append(np.mean(bits_hat != bits_test))

            # DL 解调 BER
            model.eval()
            X_test = torch.from_numpy(np.vstack([rx_test.real, rx_test.imag]).T).float().to(device)
            with torch.no_grad():
                preds = torch.argmax(model(X_test), dim=1).cpu().numpy()
            k = int(np.log2(M))
            bits_dl = (((preds[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
            ber_dl.append(np.mean(bits_dl != bits_test))

        results[M] = {'snr_db': snr_db, 'ber_traditional': ber_traditional, 'ber_dl': ber_dl}

    # ---------------------------------------------
    # 6. 绘图
    # ---------------------------------------------
    plt.figure(figsize=(8, 6))
    for M in qam_orders:
        plt.semilogy(results[M]['snr_db'], results[M]['ber_traditional'], 'o-', label=f'{M}-QAM 传统')
        plt.semilogy(results[M]['snr_db'], results[M]['ber_dl'], 's--', label=f'{M}-QAM DL')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR for Traditional vs DL QAM Demodulation (GPU)')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

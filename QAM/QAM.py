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
# 1. 参数配置
# ---------------------------------------------
# 定义符号数，即QAM阶数列表
qam_orders = [2, 4, 16, 64]  # 2-QAM其实相当于BPSK
# SNR范围（以dB为单位）
snr_db = np.arange(0, 21, 2)  # 从0到20 dB，每2 dB一个点
# 每个SNR点的符号数量
num_symbols = 100000
# 训练网络参数
learning_rate = 0.001
batch_size = 1024
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------
# 2. QAM调制函数
# ---------------------------------------------
def qam_modulate(bits, M):
    """
    对输入比特序列进行M-QAM调制
    bits: 二进制比特序列（长度必须为log2(M)的整数倍）
    M: 星座点个数，如16、64
    返回复数基带符号
    """
    k = int(np.log2(M))  # 每个符号对应的比特数
    # 将比特序列重塑为(N_symbols, k)
    bit_groups = bits.reshape(-1, k)
    # 将二进制转换为符号索引
    symbols = bit_groups.dot(1 << np.arange(k)[::-1])  # 大端
    # 生成QAM星座
    # sqrt(M)的情况，将实部和虚部分别映射到[-sqrt(M)+1 ... sqrt(M)-1]
    I = 2 * (symbols % np.sqrt(M)) - np.sqrt(M) + 1
    Q = 2 * (symbols // np.sqrt(M)) - np.sqrt(M) + 1
    return I + 1j * Q  # 复数符号

# ---------------------------------------------
# 3. QAM理想解调函数（最小欧氏距离）
# ---------------------------------------------
def qam_demodulate(rx_symbols, M):
    """
    对接收的复数符号进行最小欧氏距离解调
    rx_symbols: 接收的复数基带符号
    M: 星座点个数
    返回估计的比特序列
    """
    # 生成参考星座点表
    re = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
    im = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
    constellation = np.array([x + 1j*y for x in re for y in im])
    # 计算每个接收符号到参考星座的距离
    idx = np.argmin(abs(rx_symbols.reshape(-1, 1) - constellation.reshape(1, -1)), axis=1)
    # 将索引转换为比特
    k = int(np.log2(M))
    bits = (((idx[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
    return bits

# ---------------------------------------------
# 4. 全连接神经网络模型
# ---------------------------------------------
class QAMNet(nn.Module):
    def __init__(self, M):
        super(QAMNet, self).__init__()
        k = int(np.log2(M))  # 每个符号比特
        # 输入2维（实部+虚部），输出M分类
        self.net = nn.Sequential(
            nn.Linear(2, 128),  # 隐藏层1，128单元
            nn.ReLU(),
            nn.Linear(128, 64),  # 隐藏层2，64单元
            nn.ReLU(),
            nn.Linear(64, M)     # 输出层，M分类
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------
# 5. 主函数：仿真各QAM阶的BER性能
# ---------------------------------------------
if __name__ == '__main__':
    # 存储结果字典
    results = {}

    # 遍历不同QAM阶数
    for M in qam_orders:
        ber_traditional = []  # 传统解调BER列表
        ber_dl = []           # DL解调BER列表

        # 初始化DL模型并准备训练
        model = QAMNet(M).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 生成全部符号和标签用于训练（AWGN信道，SNR=10dB）
        # 随机比特序列
        bits = np.random.randint(0, 2, int(num_symbols * np.log2(M)))
        symbols = qam_modulate(bits, M)  # 调制
        # 添加噪声
        snr_linear = 10**(10/10)
        noise = (np.random.randn(len(symbols)) + 1j*np.random.randn(len(symbols)))
        noise *= np.sqrt(1/(2*snr_linear))
        rx = symbols + noise
        # 准备PyTorch数据集
        X = torch.from_numpy(np.vstack([rx.real, rx.imag]).T).float().to(device)
        y = torch.from_numpy((bits.reshape(-1, int(np.log2(M))).dot(
            1 << np.arange(int(np.log2(M)))[::-1]))).long().to(device)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练网络
        for epoch in range(num_epochs):
            for xb, yb in loader:
                logits = model(xb)                  # 前向
                loss = criterion(logits, yb)        # 计算损失
                optimizer.zero_grad()               # 梯度清零
                loss.backward()                     # 反向传播
                optimizer.step()                    # 更新参数

        # 对各SNR点进行测试
        for db in snr_db:
            # 计算线性SNR
            snr_lin = 10**(db/10)
            # 生成测试符号
            bits_test = np.random.randint(0, 2, int(num_symbols * np.log2(M)))
            symb_test = qam_modulate(bits_test, M)
            noise = (np.random.randn(len(symb_test)) + 1j*np.random.randn(len(symb_test)))
            noise *= np.sqrt(1/(2*snr_lin))
            rx_test = symb_test + noise

            # 传统解调
            bits_hat = qam_demodulate(rx_test, M)
            ber_traditional.append(np.mean(bits_hat != bits_test))

            # DL解调
            X_test = torch.from_numpy(np.vstack([rx_test.real, rx_test.imag]).T).float().to(device)
            with torch.no_grad():
                logits = model(X_test)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            # 将分类索引转换为比特序列
            k = int(np.log2(M))
            bits_dl = (((preds[:, None] & (1 << np.arange(k)[::-1])) > 0).astype(int)).reshape(-1)
            ber_dl.append(np.mean(bits_dl != bits_test))

        # 保存结果
        results[M] = {'snr_db': snr_db,
                      'ber_traditional': ber_traditional,
                      'ber_dl': ber_dl}

    # ---------------------------------------------
    # 6. 绘制BER vs SNR曲线
    # ---------------------------------------------
    plt.figure(figsize=(8, 6))
    for M in qam_orders:
        plt.semilogy(results[M]['snr_db'], results[M]['ber_traditional'], 'o-', label=f'{M}-QAM 传统')
        plt.semilogy(results[M]['snr_db'], results[M]['ber_dl'], 's--', label=f'{M}-QAM DL')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs SNR for Traditional vs DL QAM Demodulation')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib

# 支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class BentPipeTransponder:
    """
    模拟一次变频弯管式卫星中继载荷
    """
    def __init__(self, total_eirp_dbw, hpa_model, obo_db, ibo_db):
        self.total_eirp_dbw = total_eirp_dbw
        self.hpa = hpa_model
        self.obo_db = obo_db
        self.ibo_db = ibo_db

    def forward(self, signals):
        # 合成多用户信号
        composite = sum(signals)
        # 输入回退
        composite *= 10**(-self.ibo_db/20)
        # 非线性放大
        amplified = self.hpa.amplify(composite)
        # 输出回退
        amplified *= 10**(-self.obo_db/20)
        # 带通滤波
        bpf = signal.firwin(101, [0.2,0.8], pass_zero=False)
        filtered = signal.lfilter(bpf, [1.0], amplified)
        # EIRP归一
        return self._normalize(filtered)

    # def _normalize(self, waveform):
    #     lin_p = np.mean(np.abs(waveform)**2)
    #     target = 10**(self.total_eirp_dbw/10)
    #     return waveform * np.sqrt(target/lin_p)
    def _normalize(self, waveform):
        return waveform  # 暂时关闭归一化


class SimpleHPA:
    """行波管功放的AM/AM模型"""
    def __init__(self, sat_amp_max_dbw, sat_gain_db, alpha=2.0):
        self.Asat = 10**(sat_amp_max_dbw/10)
        self.gain = 10**(sat_gain_db/10)
        self.alpha = alpha

    def amplify(self, x):
        mag = np.abs(x)
        ym = self.gain*mag / ((1+(mag/self.Asat)**self.alpha)**(1/self.alpha))
        return ym * np.exp(1j*np.angle(x))

class Scheduler:
    """FDMA/TDMA调度"""
    def __init__(self, mode, num_users):
        self.mode = mode
        self.num_users = num_users
        self.ptr = 0

    def assign(self, signals):
        if self.mode=='FDMA':
            return signals
        out = [np.zeros_like(signals[0]) for _ in range(self.num_users)]
        active = self.ptr % self.num_users
        out[active] = signals[active]
        self.ptr += 1
        return out

class PowerControllerPID:
    """基于C/N0反馈的PID"""
    def __init__(self, Kp, Ki, Kd, target):
        self.Kp,self.Ki,self.Kd = Kp,Ki,Kd
        self.target = target
        self.integral = None
        self.prev = None

    def init_state(self, N):
        self.integral = np.zeros(N)
        self.prev = np.zeros(N)

    def update(self, meas):
        err = self.target - np.array(meas)
        self.integral += err
        deriv = err - self.prev
        self.prev = err
        return self.Kp*err + self.Ki*self.integral + self.Kd*deriv

if __name__=='__main__':
    # 参数
    N = 3
    total_eirp = 50    # 卫星发射EIRP (dBW)
    bw = 36e6          # Hz
    target_cn0 = 20    # dBW
    gt_t = [10,10,10]

    # 初始化模块
    hpa = SimpleHPA(47, 10)
    trans = BentPipeTransponder(total_eirp, hpa, obo_db=3, ibo_db=3)
    sched_fdma = Scheduler('FDMA', N)
    sched_tdma = Scheduler('TDMA', N)
    pid_f = PowerControllerPID(0.1,0.01,0.05,target_cn0)
    pid_t = PowerControllerPID(0.1,0.01,0.05,target_cn0)
    pid_f.init_state(N)
    pid_t.init_state(N)

    # 发射功率增益状态 (dB)
    gain_f = np.zeros(N)
    gain_t = np.zeros(N)

    # 历史记录
    dist_h = []
    cn0_fh = [[] for _ in range(N)]
    cn0_th = [[] for _ in range(N)]
    dp_f = [[] for _ in range(N)]
    dp_t = [[] for _ in range(N)]
    cap_f = []
    cap_t = []

    for frame in range(100):
        # 动态位置
        d = 36000 + np.random.uniform(-10,10,N)
        dist_h.append(d)
        L = 92.45 + 20*np.log10(d) + 20*np.log10(12)

        # 生成QPSK信号并作用发射增益
        sigs = []
        for i in range(N):
            base = np.random.choice([1+1j,1-1j,-1+1j,-1-1j], 1024)
            sigs.append(base * 10**(gain_f[i]/20))  # FDMA 使用 gain_f

        # FDMA
        out_f = trans.forward(sched_fdma.assign(sigs))
        pr = 10*np.log10(np.mean(np.abs(out_f)**2))
        noise = -228.6 + 10*np.log10(bw)
        cn0_f = [pr - noise - L[i] + gt_t[i] for i in range(N)]
        dp = pid_f.update(cn0_f)
        gain_f += dp  # 更新发射增益
        for i in range(N):
            dp_f[i].append(dp[i])
            cn0_fh[i].append(cn0_f[i])
        snr = [10**(c/10) for c in cn0_f]
        cap_f.append(sum([(bw/N)*np.log2(1+s) for s in snr])/1e6)

        # 为TDMA重新生成信号
        sigs = []
        for i in range(N):
            base = np.random.choice([1+1j,1-1j,-1+1j,-1-1j], 1024)
            sigs.append(base * 10**(gain_t[i]/20))
        # TDMA
        out_t = trans.forward(sched_tdma.assign(sigs))
        pr = 10*np.log10(np.mean(np.abs(out_t)**2))
        cn0_t = [pr - noise - L[i] + gt_t[i] for i in range(N)]
        dp = pid_t.update(cn0_t)
        gain_t += dp  # 更新发射增益
        for i in range(N):
            dp_t[i].append(dp[i])
            cn0_th[i].append(cn0_t[i])
        snr = [10**(c/10) for c in cn0_t]
        cap_t.append(np.mean([bw*np.log2(1+s) for s in snr])/1e6)

    # # 绘图函数
    # def plot_series(data, title, ylabel):
    #     plt.figure(figsize=(8,4))
    #     for i in range(N): plt.plot(data[i], label=f'用户{i+1}')
    #     plt.title(title)
    #     plt.xlabel('帧数')
    #     plt.ylabel(ylabel)
    #     plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # plot_series(dp_f, 'PID ΔP (FDMA)', 'ΔP (dB)')
    # plot_series(dp_t, 'PID ΔP (TDMA)', 'ΔP (dB)')
    # plot_series(cn0_fh, 'C/N0 (FDMA)', 'C/N0 (dBW)')
    # plot_series(cn0_th, 'C/N0 (TDMA)', 'C/N0 (dBW)')

    # plt.figure(figsize=(8,4))
    # for i in range(N): plt.plot([h[i] for h in dist_h], label=f'用户{i+1}')
    # plt.title('用户距离变化')
    # plt.xlabel('帧数')
    # plt.ylabel('距离 (km)')
    # plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 多图合并绘制
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# FDMA ΔP
for i in range(N):
    axs[0, 0].plot(dp_f[i], label=f'用户{i+1}')
axs[0, 0].set_title('PID ΔP (FDMA)')
axs[0, 0].set_xlabel('帧数')
axs[0, 0].set_ylabel('ΔP (dB)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# TDMA ΔP
for i in range(N):
    axs[0, 1].plot(dp_t[i], label=f'用户{i+1}')
axs[0, 1].set_title('PID ΔP (TDMA)')
axs[0, 1].set_xlabel('帧数')
axs[0, 1].set_ylabel('ΔP (dB)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# FDMA C/N0
for i in range(N):
    axs[1, 0].plot(cn0_fh[i], label=f'用户{i+1}')
axs[1, 0].set_title('C/N0 (FDMA)')
axs[1, 0].set_xlabel('帧数')
axs[1, 0].set_ylabel('C/N0 (dBW)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# TDMA C/N0
for i in range(N):
    axs[1, 1].plot(cn0_th[i], label=f'用户{i+1}')
axs[1, 1].set_title('C/N0 (TDMA)')
axs[1, 1].set_xlabel('帧数')
axs[1, 1].set_ylabel('C/N0 (dBW)')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# 用户距离变化单独画
plt.figure(figsize=(8, 4))
for i in range(N):
    plt.plot([h[i] for h in dist_h], label=f'用户{i+1}')
plt.title('用户距离变化')
plt.xlabel('帧数')
plt.ylabel('距离 (km)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"均等共享EIRP: {total_eirp-10*np.log10(N):.2f} dBW")
print(f"FDMA 平均容量: {np.mean(cap_f):.2f} Mbps")
print(f"TDMA 平均容量: {np.mean(cap_t):.2f} Mbps")
print(f"[Frame {frame}] FDMA gain_f: {gain_f}")
print(f"[Frame {frame}] FDMA cn0: {cn0_f}")


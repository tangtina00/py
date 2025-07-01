# 初始化网络 Q(s,a)
Q = DQN()  
num_episodes = 1000  # 你可以根据需要设置训练轮数
for episode in range(num_episodes):
    state = get_current_state()  # 各用户C/N0向量
    terminal = False
    while not terminal:
        action = select_action(Q, state)  # 输出各用户功率调节量
        apply_action(action)              # 调整功率
        next_state = observe_state()      # 获得新状态（C/N0）
        # 奖励 = 当前所有用户中的最低 C/N0
        reward = min(next_state.CN0_list)
        store_transition(state, action, reward, next_state)
        train_DQN()
        state = next_state
        # 根据实际情况设置终止条件
        terminal = check_terminal_condition(next_state)

import copy
import numpy as np

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0]* self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma

        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]
    
    def value_iteration(self):
        cnt = 0
        while(True):
            max_diff = 0
            new_v = np.zeros([self.env.ncol*self.env.nrow])
            #each routine will update full V(s)
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    #calculate q function for each available next state
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma*self.v[next_state] * (1-done))
                    qsa_list.append(qsa)
                #then choose the max one(which means a best action and a best policy)
                new_v[s] = max(qsa_list)
                max_diff_candidate = abs(new_v[s] - self.v[s])
                if(max_diff < max_diff_candidate):
                    max_diff = max_diff_candidate
            
            self.v = new_v
            if(max_diff < self.theta):
                break
            cnt += 1
        print("value iteration共进行了%d轮" %cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略(永远前往下一步V最大的地方)
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
    
class PolicyIteration:
    def __init__(self,env,theta,gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = np.zeros([self.env.nrow * self.env.ncol])
        #初始策略：完全随机 脑血栓走法
        self.pi = [[0.25,0.25,0.25,0.25] for i in range(self.env.ncol * self.env.nrow)]

    def policy_evaluation(self):
        cnt = 0
        while(True):
            max_diff = 0
            new_v = np.zeros([self.env.ncol*self.env.nrow])
            #each routine will update full V(s)
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    #calculate q function for each available next state
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma*self.v[next_state] * (1-done))
                    qsa_list.append(qsa * self.pi[s][a])
                #then SUM UP all reward based on current state and current policy
                #so it's called "evaluation of V"
                new_v[s] = sum(qsa_list)
                max_diff_candidate = abs(new_v[s] - self.v[s])
                if(max_diff < max_diff_candidate):
                    max_diff = max_diff_candidate
            
            self.v = new_v
            if(max_diff < self.theta):
                print("策略评估在%d轮后完成" %cnt)
                break
            cnt += 1

    def policy_improvement(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完毕")
        return self.pi
    
    def policy_iteration(self):
        while(True):
            old_pi = copy.deepcopy(self.pi)
            self.policy_evaluation()
            new_pi = self.policy_improvement()
            if(old_pi == new_pi):
                print("收敛！")
                break
    

env = CliffWalkingEnv()
action_meaning = ['↑','↓','←','→']
theta = 0.001
gamma = 0.9
# agent = ValueIteration(env,theta,gamma)
# agent.value_iteration()
agent = PolicyIteration(env,theta,gamma)
agent.policy_iteration()

print_agent(agent,action_meaning,list(range(37,47)),[47])
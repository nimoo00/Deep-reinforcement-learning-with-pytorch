import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class QLearningAgent:
    def __init__(self):
        # 하이퍼파라미터 및 환경 설정
        self.alpha = 0.1           # 학습률
        self.gamma = 0.95          # 할인 계수
        self.epsilion = 0.9        # 탐욕 정책에서 무작위 선택 확률
        self.n_state = 20          # 환경의 상태 개수
        self.actions = ['left', 'right']  # 가능한 행동들
        self.max_episodes = 20    # 에피소드 수
        self.fresh_time = 0.1      # 환경 업데이트 시 대기 시간 (초)
        
        # Q-테이블과 에피소드별 스텝 수 기록
        self.q_table = self.build_q_table(self.n_state, self.actions)
        self.step_counter_times = []
        # self.reward_history = []
    def build_q_table(self, n_state, actions):
        """
        초기 Q-테이블 생성: n_state x len(actions) 크기의 0으로 초기화된 DataFrame 반환
        """
        q_table = pd.DataFrame(
            np.zeros((n_state, len(actions))),
            index=np.arange(n_state),
            columns=actions
        )
        return q_table

    def choose_action(self, state):
        """
        ε-greedy 정책으로 현재 상태에서 행동 선택
        """
        state_action = self.q_table.loc[state, :]
        if np.random.uniform() > self.epsilion or (state_action == 0).all():
            action_name = np.random.choice(self.actions)
        else:
            action_name = state_action.idxmax()
        return action_name

    def get_env_feedback(self, state, action):
        """
        현재 상태와 행동에 따른 환경의 반응(다음 상태와 보상) 결정
        """
        if action == 'right':
            if state == self.n_state - 2:
                next_state = 'terminal'
                reward = 1
            else:
                next_state = state + 1
                reward = -0.5
        else:  # action == 'left'
            if state == 0:
                next_state = 0
            else:
                next_state = state - 1
            reward = -0.5
        return next_state, reward

    def update_env(self, state, episode, step_counter):
        """
        현재 상태를 텍스트 기반으로 출력하여 환경의 진행 상황을 시각적으로 보여줌.
        상태가 'terminal'이면 에피소드 종료 메시지를 출력.
        """
        env = ['-'] * (self.n_state - 1) + ['T']
        if state == 'terminal':
            print("Episode {}, the total step is {}".format(episode + 1, step_counter))
            return True, step_counter
        else:
            env[state] = '*'
            env_str = ''.join(env)
            print(env_str)
            time.sleep(self.fresh_time)
            return False, step_counter

    def q_learning(self):
        """
        Q-러닝 알고리즘을 실행하여 최종 Q-테이블과 에피소드별 스텝 수를 반환.
        """
        for episode in range(self.max_episodes):
            state = 0
            is_terminal = False
            step_counter = 0
            self.update_env(state, episode, step_counter)
            while not is_terminal:
                action = self.choose_action(state)
                next_state, reward = self.get_env_feedback(state, action)
                
                # 터미널 상태라면 Q 업데이트 없이 보상만 사용
                if next_state == 'terminal':
                    is_terminal = True
                    q_target = reward
                else:
                    # Q-러닝 업데이트 공식 적용
                    delta = reward + self.gamma * self.q_table.iloc[next_state, :].max() - self.q_table.loc[state, action]
                    self.q_table.loc[state, action] += self.alpha * delta
                
                state = next_state
                is_terminal, steps = self.update_env(state, episode, step_counter + 1)
                step_counter += 1
                self.reward_history.append([step_counter,reward])
                # if next_state != 'terminal' and abs(delta) <= 0.001:
                #     is_terminal = True
                # if is_terminal:
                #     self.step_counter_times.append(steps)
                #     with open("reward_history.txt","a") as f:
                #         for i in self.reward_history:
                #             f.write(f"{i}\n")
                #     self.reward_history = []
        return self.q_table, self.step_counter_times

    def run(self):
        """
        Q-러닝 실행 후, 최종 Q-테이블과 에피소드별 스텝 수를 출력하고 그래프로 시각화.
        """
        q_table, step_counter_times = self.q_learning()
        print("Q table\n{}\n".format(q_table))
        print('end')
        plt.plot(step_counter_times, 'g-')
        plt.ylabel("steps")
        plt.xlabel("Episode")
        plt.show()
        print("The step_counter_times is {}".format(step_counter_times))


if __name__ == '__main__':
    agent = QLearningAgent()
    agent.run()

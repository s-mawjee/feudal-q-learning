"""
Feudal RL Agent
"""

import numpy as np
import pandas as pd


class FeudalQLearningTable:
    def __init__(self, actions):
        self.levels = {
            0: FeudalLevel(actions=[4]),
            1: FeudalLevel(actions=list(range(5))),
            2: FeudalLevel(actions=actions)
        }

    def choose_action(self, state):
        level_0_state, level_1_state, level_2_state = self.get_level_states(
            state)
        actions = [-1, -1, -1]
        actions[0] = self.levels[0].choose_action(level_0_state)
        actions[1] = self.levels[1].choose_action(level_1_state)
        actions[2] = self.levels[2].choose_action(level_2_state)
        level = 0
        if actions[0] == 4:
            level = 1
            if actions[1] == 4:
                level = 2

        return actions, level

    def learn(self, s, actions, r, s_, done):
        level_0_s, level_1_s, level_2_s = self.get_level_states(s)
        level_0_s_, level_1_s_, level_2_s_ = self.get_level_states(s_)

        obey_reward = 0
        not_obey_reward = -1

        self.levels[0].learn(level_0_s, actions[0], r, level_0_s_, done)
        if actions[0] == 4:
            self.levels[1].learn(level_1_s, actions[1], r, level_1_s_, done)
            if actions[1] == 4:
                self.levels[2].learn(
                    level_2_s, actions[2], r, level_2_s_, done)
            else:
                if actions[2] == actions[1]:
                    self.levels[2].learn(
                        level_2_s, actions[2], obey_reward, level_2_s_, done)
                else:
                    self.levels[2].learn(
                        level_2_s, actions[2], not_obey_reward, level_2_s_, done)
        else:
            if actions[1] == actions[0]:
                self.levels[1].learn(level_1_s, actions[1],
                                     obey_reward, level_1_s_, done)
            else:
                self.levels[1].learn(level_1_s, actions[1],
                                     not_obey_reward, level_1_s_, done)

    @staticmethod
    def get_level_states(state):
        level_2_state = state
        level_1_state = (int(level_2_state[0] / 2), int(level_2_state[1] / 2))
        level_0_state = (int(level_1_state[0] / 2), int(level_1_state[1] / 2))

        return level_0_state, level_1_state, level_2_state


class FeudalLevel:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        observation = str(observation)
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

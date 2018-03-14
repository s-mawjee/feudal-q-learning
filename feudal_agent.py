"""
Feudal RL Agent
"""

import numpy as np
import pandas as pd


class FeudalQLearningTable:
    def __init__(self, numberActions, numberLayers=3):
        self.number_actions = numberActions
        self.number_layers = numberLayers
        self.levels = {0: FeudalLevel(actions=[numberActions])}
        if(numberLayers > 1):
            for i in range(1, numberLayers-1):
                self.levels[i] = FeudalLevel(
                    actions=list(range(numberActions+1)))
            self.levels[numberLayers -
                        1] = FeudalLevel(actions=list(range(numberActions)))

    def choose_action(self, state):
        level_states = self.get_level_states(state)
        actions = []
        for a in range(self.number_layers):
            actions.append(self.levels[a].choose_action(level_states[a]))
        return actions

    def learn(self, s, actions, r, s_, done):
        level_states = self.get_level_states(s)
        level_states_prime = self.get_level_states(s_)

        obey_reward = 0
        not_obey_reward = -1

        for i in range(self.number_layers):
            if i == 0:
                reward = r
            else:
                if actions[i-1] == 4:
                    reward = r
                else:
                    if actions[i-1] == actions[i]:
                        reward = obey_reward
                    else:
                        reward = not_obey_reward

            self.levels[i].learn(level_states[i], actions[i],
                                 reward, level_states_prime[i], done)

    def get_level_states(self, state):
        states = []
        states.append(state)
        for i in range(self.number_layers-2, -1, -1):
            states.append((int(states[-1][0]/2), int(states[-1][1]/2)))
        states.reverse()
        return states


class FeudalLevel:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float_)

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

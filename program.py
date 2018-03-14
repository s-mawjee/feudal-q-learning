"""
Feudal Q learning

"""
import pandas as pd
from feudal_agent import FeudalQLearningTable
from maze_env import Maze


def update():
    for episode in range(100):
        episode_log = pd.DataFrame(
            columns=['ACTION', 'REWARD', 'A0', 'A1', 'A2', 'A3', 'V'])
        step_counter = 1
        episode_reward = 0
        observation = env.reset()
        while True:
            env.render()

            actions, level = rl.choose_action(observation)

            observation_, reward, done = env.step(actions[2])

            episode_reward += reward

            rl.learn(observation, actions, reward, observation_, done)
            print(str(episode + 1) + "-" + str(step_counter),
                  str(observation), actions, str(observation_))

            action_values = rl.levels[2].q_table.loc[str(observation), :]
            episode_log.loc[len(episode_log)] = [actions[2], reward, action_values[0], action_values[1],
                                                 action_values[2], action_values[3], episode_reward]

            observation = observation_
            if done:
                break
            step_counter = step_counter + 1

        # episode_log = episode_log.set_index('ACTION')
        episode_log.head()
        name = 'Analysis/frames/log' + str(episode + 1) + '.csv'
        episode_log.to_csv(name, encoding='utf-8')

        log_name = './Analysis/frames/log' + str(episode + 1) + '.csv'
        log_gif = './Analysis/frames/image' + str(episode + 1) + '.gif'

        log.loc[len(log)] = [episode + 1, step_counter,
                             episode_reward, log_gif, log_name]

    print('Game Over')
    env.destroy()


if __name__ == "__main__":
    log = pd.DataFrame(columns=['Episode', 'Length', 'Reward', 'IMG', 'LOG'])
    env = Maze()
    rl = FeudalQLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
    log = log.set_index('Episode')
    log.to_csv('./Analysis/log.csv', encoding='utf-8')
    #rl.levels[0].q_table.to_csv('level_0.csv', encoding='utf-8')
    #rl.levels[1].q_table.to_csv('level_1.csv', encoding='utf-8')
    #rl.levels[2].q_table.to_csv('level_2.csv', encoding='utf-8')

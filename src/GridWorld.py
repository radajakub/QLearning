import numpy as np


class GridWorld:
    def __init__(self, gamma=0.99):
        # Set information about the gridworld
        self.gamma = gamma
        self.row_max = 10
        self.col_max = 10
        self.grid = np.zeros((self.row_max, self.col_max))

        # Set initial location (lower left corner state)
        self.agent_location = (self.row_max - 1, 0)

        # Set terminal locations : trap points & goal point (upper right corner state)
        self.trap_location1 = (2, self.col_max - 1)
        self.trap_location2 = (2, self.col_max - 2)
        self.trap_location3 = (2, self.col_max - 3)
        self.trap_location4 = (2, self.col_max - 4)
        self.trap_location5 = (1, self.col_max - 3)
        self.trap_location6 = (0, 0)
        self.trap_location7 = (1, 1)
        self.trap_location8 = (2, 2)
        self.trap_location9 = (3, 3)
        self.trap_location10 = (4, 4)
        self.trap_location11 = (4, 6)
        self.trap_location12 = (6, 4)
        self.goal_location = (0, self.col_max - 1)
        self.terminal_states = [self.trap_location1, self.trap_location2, self.trap_location3, self.trap_location4,
                                self.trap_location5, self.trap_location6, self.trap_location7, self.trap_location8,
                                self.trap_location9, self.trap_location10, self.trap_location11, self.trap_location12,
                                self.goal_location]

        # Set actions & initial return
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.return_value = 0

    def get_reward(self):
        if self.agent_location in self.terminal_states:
            reward = 100 if self.agent_location == self.terminal_states[-1] else -10
            terminal = True
        else:
            reward = -1
            terminal = False
        self.return_value = reward + self.gamma * self.return_value

        return reward, terminal

    def make_step(self, action):
        # introduce stochastic transition
        if np.random.uniform(0, 1) < 0.25:
            action = self.actions[np.random.randint(0, len(self.actions))]

        # interact with the gridworld
        if action == 'UP':
            if self.agent_location[0] == 0:
                reward, terminal = self.get_reward()
            else:
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
                reward, terminal = self.get_reward()
        elif action == 'DOWN':
            if self.agent_location[0] == self.row_max - 1:
                reward, terminal = self.get_reward()
            else:
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])
                reward, terminal = self.get_reward()
        elif action == 'LEFT':
            if self.agent_location[1] == 0:
                reward, terminal = self.get_reward()
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
                reward, terminal = self.get_reward()
        elif action == 'RIGHT':
            if self.agent_location[1] == self.col_max - 1:
                reward, terminal = self.get_reward()
            else:
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
                reward, terminal = self.get_reward()

        return self.agent_location, reward, terminal

    def reset(self):
        self.agent_location = (self.row_max - 1, 0)
        episode_return = self.return_value
        self.return_value = 0

        return episode_return




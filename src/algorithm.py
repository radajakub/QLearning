import numpy as np
from collections import defaultdict
from random import random, randint


class Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.Q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions  # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max

    def get_Q_table(self):
        return self.Q_table

    def action(self, state):
        if random() < self.epsilon:
            # sample random action index with probability epsilon
            action_index = randint(0, len(self.actions) - 1)
        else:
            # otherwise select the greedy action w.r.t. self.Q_table
            action_index = np.argmax(self.Q_table[state])
        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        # get index of action
        current_action = self.actions.index(action)
        # Q(s,a) <- Q(s, a) + alpha * (r + gamma * max Q(s',a') - Q(s, a))
        self.Q_table[current_state][current_action] += self.alpha * (reward + self.gamma * np.max(
            self.Q_table[next_state]) - self.Q_table[current_state][current_action])

    def get_max_Q_function(self):
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))

        for row, col in np.ndindex(max_Q_table.shape):
            max_Q_table[row, col] = np.max(self.Q_table[(row, col)])

        return max_Q_table


class Double_Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.Q1 = defaultdict(lambda: np.zeros(4))
        self.Q2 = defaultdict(lambda: np.zeros(4))
        self.Q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions  # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max

    def get_Q_table(self):
        return self.Q_table

    def action(self, state):
        if random() < self.epsilon:
            # sample random action index [0, 3] with probability epsilon
            action_index = randint(0, len(self.actions) - 1)
        else:
            # otherwise select the greedy action w.r.t. self.Q_table
            action_index = np.argmax(self.Q_table[state])

        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        current_action = self.actions.index(action)

        if random() < 0.5:
            # Q1(s, a) <- Q1(s, a) + alpha * (r + gamma * Q2(s', argmax_a' Q1(s', a')) - Q1(s, a))
            self.Q1[current_state][current_action] += self.alpha * (reward + self.gamma * self.Q2[
                next_state][np.argmax(self.Q1[next_state])] - self.Q1[current_state][current_action])
        else:
            # Q2(s, a) <- Q2(s, a) + alpha * (r + gamma * Q1(s', argmax_a' Q2(s', a')) - Q2(s, a))
            self.Q2[current_state][current_action] += self.alpha * (reward + self.gamma * self.Q1[
                next_state][np.argmax(self.Q2[next_state])] - self.Q2[current_state][current_action])

        # compute Q-table as Q1 + Q2 for given state and action
        self.Q_table[current_state][current_action] = self.Q1[current_state][current_action] + \
            self.Q2[current_state][current_action]

    def get_max_Q_function(self):
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))

        for row, col in np.ndindex(max_Q_table.shape):
            max_Q_table[row, col] = np.max(self.Q_table[(row, col)])

        return max_Q_table

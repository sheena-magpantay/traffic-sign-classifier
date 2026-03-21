import numpy as np
import random

class RLAgent:
    def __init__(self, num_states, num_actions):
        self.q_table = np.zeros((num_states, num_actions))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.q_table.shape[1]-1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])

        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )



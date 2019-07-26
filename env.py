from typing import List

import gym
import numpy as np
from gym import spaces


class WritingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'text']}

    N_DISCRETE_ACTIONS = 6

    def __init__(self, width=3, height=3):
        super(WritingEnvironment, self).__init__()

        self.height = height
        self.width = width
        self.board_shape = (height, width)
        self.board_pos: np.ndarray = np.zeros(2)
        self.board: np.ndarray = np.zeros(self.board_shape)
        self.pattern: np.ndarray = np.zeros(self.board_shape)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(WritingEnvironment.N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=1, shape=(height, width), dtype=np.uint8)
        self.observation_space = spaces.Discrete(3 * height * width)

        self.reset()

    @property
    def state(self):
        pos = np.zeros(self.board_shape)
        row, col = self.board_pos
        pos[row, col] = 1

        return np.concatenate((self.board.flatten(), self.pattern.flatten(), pos.flatten()))

    def step(self, action):
        move_reward = -1
        correct_square_reward = 2
        incorrect_square_reward = -2
        bad_quit_reward = -1000
        good_quit_reward = 1

        done = False
        info = dict()

        if action == 0:
            # Move Up
            self.board_pos[0] = max(self.board_pos[0] - 1, 0)
            reward = move_reward
        elif action == 1:
            # Move Right
            self.board_pos[1] = min(self.board_pos[1] + 1, self.width - 1)
            reward = move_reward
        elif action == 2:
            # Move Down
            self.board_pos[0] = min(self.board_pos[0] + 1, self.height - 1)
            reward = move_reward
        elif action == 3:
            # Move Left
            self.board_pos[1] = max(self.board_pos[1] - 1, 0)
            reward = move_reward
        elif action == 4:
            row, col = self.board_pos
            self.board[row, col] = 1

            if self.pattern[row, col] == 1:
                reward = correct_square_reward
            else:
                reward = incorrect_square_reward
        elif action == 5:
            reward = good_quit_reward if np.array_equal(self.board, self.pattern) else bad_quit_reward
            done = True
        else:
            raise ValueError('Unrecognised action: %s' % str(action))

        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = np.zeros(self.board_shape)
        self.board_pos = np.array([self.height // 2, self.width // 2])
        self.pattern = np.array([[1, 1, 1],
                                 [0, 1, 0],
                                 [0, 1, 0]])

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if mode == 'text':
            self._render_text()

    def _render_text(self):
        print('―' * 2 * (self.width + 2))

        for row in range(self.height):
            board_string = '|'
            pattern_string = '|'

            for col in range(self.width):
                cell_val = self.board[row, col]

                if cell_val == 0:
                    shirushi = ' '
                elif cell_val == 1 == self.pattern[row, col]:
                    shirushi = 'o'
                else:
                    shirushi = 'x'

                board_string += '%s' % shirushi
                pattern_string += 'o' if self.pattern[row, col] else ' '

            board_string += '|'
            pattern_string += '|'
            print(pattern_string + board_string)

        print('―' * 2 * (self.width + 2))


if __name__ == '__main__':
    env = WritingEnvironment()
    state = env.reset()

    for step in range(10):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        print('Step %02d - Reward: %d' % (step + 1, reward))
        env.render(mode='text')

        if done:
            break

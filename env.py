from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple

import gym
import numpy as np
import pyglet
from gym import spaces
from gym.envs.classic_control import rendering
from pyglet.window import key

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
FILL_SQUARE = 4
QUIT = 5


class KeyStateHandler:
    """Simple handler that tracks the state of keys on the keyboard. If a
    key is pressed then this handler holds a True value for it.

    For example::

        >>> win = pyglet.window.Window
        >>> keyboard = KeyStateHandler()
        >>> win.push_handlers(keyboard)

        # Hold down the "up" arrow...

        >>> keyboard[key.UP]
        True
        >>> keyboard.key_was_pressed(key.UP)
        True
        >>> keyboard.key_was_released(key.UP)
        False
        >>> keyboard.key_was_held(key.UP)
        True
        >>> keyboard[key.DOWN]
        False

    """

    def __init__(self):
        self.curr_state = defaultdict(lambda: False)
        self.prev_state = defaultdict(lambda: False)

    def on_key_press(self, symbol, modifiers):
        self.prev_state[symbol] = self.curr_state[symbol]
        self.curr_state[symbol] = True

    def on_key_release(self, symbol, modifiers):
        self.prev_state[symbol] = self.curr_state[symbol]
        self.curr_state[symbol] = False

    def key_was_pressed(self, symbol):
        return not self.prev_state[symbol] and self.curr_state[symbol]

    def key_was_released(self, symbol):
        return self.prev_state[symbol] and not self.curr_state[symbol]

    def key_was_held(self, symbol):
        return self.prev_state[symbol] and self.curr_state[symbol]

    def __getitem__(self, key):
        return self.curr_state[key]


class WritingEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    metadata = {'render.modes': ['human', 'text']}

    N_DISCRETE_ACTIONS = 6

    def __init__(self, width=3, height=3, cell_size=80):
        super(WritingEnvironment, self).__init__()

        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.board_shape = (height, width)
        self.agent_position: np.ndarray = np.zeros(2)
        self.board: np.ndarray = np.zeros(self.board_shape)
        self.pattern: np.ndarray = np.zeros(self.board_shape)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(WritingEnvironment.N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, 3), dtype=np.uint8)
        # self.observation_space = spaces.Discrete(3 * height * width)

        self.viewer: Optional[rendering.Viewer] = None

    @property
    def state(self):
        pos = np.zeros(self.board_shape)
        row, col = self.agent_position
        pos[row, col] = 1

        return np.stack((self.board, self.pattern, pos))

    @property
    def should_quit(self):
        return self.viewer and \
               (not self.viewer.isopen or self.keys.key_was_pressed(key.ESCAPE) or self.keys.key_was_pressed(key.Q))

    @property
    def should_start(self):
        return self.viewer.isopen and self.keys.key_was_pressed(key.SPACE) or self.keys.key_was_pressed(key.ENTER)

    def _move(self, direction) -> bool:
        row, col = self.agent_position

        if direction == MOVE_UP:
            row -= 1  # Visually, 2d arrays start at the top-left corner and rows increase in number as they go down.
        elif direction == MOVE_DOWN:
            row += 1
        elif direction == MOVE_LEFT:
            col -= 1
        elif direction == MOVE_RIGHT:
            col += 1
        else:
            raise ValueError('Unrecognised direction \'%d\'.' % direction)

        new_pos = (row, col)

        if self._is_position_valid(new_pos):
            self.agent_position = new_pos
            return True
        else:
            return False

    def _is_position_valid(self, pos):
        row, col = pos

        return 0 <= row < self.height and 0 <= col < self.width

    def step(self, action: int):
        move_reward = -1
        correct_square_reward = 2
        incorrect_square_reward = -2
        bad_end = -1000
        good_end = 1

        done = False
        info = dict()

        if action == FILL_SQUARE:
            row, col = self.agent_position

            # Agent shouldn't try to fill the same cell twice.
            if self.board[row, col]:
                reward = incorrect_square_reward
            else:
                self.board[row, col] = 1

                # Agent should only fill in cells that are also filled in the reference pattern
                if self.pattern[row, col] == 1:
                    reward = correct_square_reward
                else:
                    reward = incorrect_square_reward
        elif action == QUIT:
            # Agent should only quit when the pattern has been copied exactly.
            reward = good_end if np.array_equal(self.board, self.pattern) else bad_end
            done = True
        elif 0 <= action < WritingEnvironment.N_DISCRETE_ACTIONS:
            # Agent should only move within the defined grid world.
            if self._move(action):
                reward = move_reward
            else:
                reward = bad_end
                done = True
        else:
            raise ValueError('Unrecognised action: %s' % str(action))

        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board = np.zeros(self.board_shape)
        self.agent_position = np.array([self.height // 2, self.width // 2])
        self.pattern = np.array([[1, 1, 1],
                                 [0, 1, 0],
                                 [0, 1, 0]])

        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if mode == 'text':
            self._render_text()
        elif mode in {'human', 'rgb_array'}:
            return self._render(mode)
        else:
            raise NotImplementedError

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

    def _render(self, mode='human'):
        """Render the environment to a window.

        Code adapted from: https://github.com/openai/gym/blob/788524a66b0e8dba37bfce6885be29fee1f8c73e/gym/envs/box2d/car_racing.py#L343

        :return:
        """
        if self.viewer is None:
            self.viewer = rendering.Viewer(WritingEnvironment.WINDOW_WIDTH,
                                           WritingEnvironment.WINDOW_HEIGHT)

            pyglet.gl.glClearColor(1, 1, 1, 1)
            self.keys = KeyStateHandler()
            self.viewer.window.push_handlers(self.keys)

        self._draw_state()

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _draw_state(self):
        self._draw_board(self.pattern, (20, WritingEnvironment.WINDOW_HEIGHT - 20))
        self._draw_board(self.board, (WritingEnvironment.WINDOW_WIDTH // 2 + 20, WritingEnvironment.WINDOW_HEIGHT - 20),
                         draw_position_marker=True)

    def _draw_board(self, board: np.ndarray, origin: Tuple[int, int], draw_position_marker: bool = False):
        """Draw a board.

        :param board: The board to draw.
        :param origin: The pixel coordinates of the top left corner of where to draw the board.
        :param draw_position_marker: Whether or not to draw the agent's position marker onto the board.
        """
        x, y = origin

        for row in range(self.width):
            for col in range(self.height):
                self._draw_square(x + col * self.cell_size, y - row * self.cell_size, filled=board[row, col] > 0)

        if draw_position_marker:
            row, col = self.agent_position
            self._draw_position_marker(x + col * self.cell_size, y - (row + 1) * self.cell_size)

    def _draw_square(self, x, y, filled=True):
        bl = [x, y]
        br = [x + self.cell_size, y]
        tl = [x, y - self.cell_size]
        tr = [x + self.cell_size, y - self.cell_size]

        self.viewer.draw_polygon([bl, tl, tr, br], filled, color=(0, 0, 0))

    def _draw_position_marker(self, x, y, res=16, filled=True):
        radius = self.cell_size / 4
        offset = self.cell_size // 2

        geom = self.viewer.draw_circle(radius, res, filled, color=(255, 0, 0))

        for i, point in enumerate(geom.v):
            geom.v[i] = (point[0] + x + offset, point[1] + y + offset)

    def wait(self, duration):
        """Essentially perform a no-op while still processing GUI events.

        :returns: False is window was closed during wait, True otherwise.
        """
        duration = timedelta(seconds=duration)
        start = datetime.now()

        self.viewer.window.dispatch_events()
        delta = timedelta()

        while delta < duration:
            if self.should_quit:
                return False

            self.viewer.window.dispatch_events()
            delta = datetime.now() - start

        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = WritingEnvironment()
    isopen = True
    episode = 0

    while isopen:
        episode += 1
        state = env.reset()

        for step in range(10):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)

            print('Episode %02d - Step %02d - Reward: %d' % (episode, step + 1, reward))
            env.render()
            isopen = env.wait(1.0)

            if done or not isopen:
                break

        while True:
            env.wait(0)

            if env.should_quit:
                isopen = False
                break
            elif env.should_start:
                break

    env.close()

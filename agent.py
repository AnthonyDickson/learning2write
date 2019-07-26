import math
import random
from abc import ABC, abstractmethod
from datetime import datetime
from statistics import mean
from typing import Tuple, List

import gym
import numpy as np
import tensorflow as tf


class ModelI(tf.keras.Model, ABC):
    @abstractmethod
    def action_value(self, state) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


class A2C:
    def __init__(self, model, env: gym.Env,
                 epsilon_max=0.9, epsilon_min=0.05, epsilon_decay=100,
                 random_state=None):
        if random_state:
            tf.random.set_seed(random_state)
            random.seed(random_state)

        self.model = model
        self.optimiser = tf.keras.optimizers.RMSprop()
        # TODO: self.model.compile(..)
        self.env = env

        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max
        self.step = 0

    def get_action(self, state):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                       math.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1

        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            action, _ = self.model.action_value(state.reshape(1, -1))

            return np.argmax(action)

    def train(self, updates=1000, max_steps=100):
        """Train the model.

        :param updates: Total number of steps to train for.
        :param max_steps: Maximum number of steps per episode.
        """
        update = 0
        episode = 0
        start = datetime.now()

        while update < updates:
            state = self.env.reset()
            states = [state]
            actions = []
            rewards = []
            step = 0

            while step < max_steps and update < updates:
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                # TODO: self.buffer.add(state, action, reward)
                # TODO: self.update() -> self.model.train(a_batch)

                self._print_progress(episode, step, update, updates, rewards, start, end='')

                step += 1
                update += 1

                if done:
                    break

            self._print_progress(episode, step, update, updates, rewards, start)
            self.env.render(mode='text')

            episode += 1

    @staticmethod
    def _print_progress(episode: int, step: int, update: int, updates: int, rewards: List[int], start: datetime,
                        end='\n'):
        elapsed_time = datetime.now() - start

        print('\rUpdate: %03d/%03d - Episode %03d - Step: %03d - '
              'Reward: %d -  Avg. Reward: %.2f - Cumulative Reward: %d - '
              'Elapsed Time: %s - Avg. Time per Update: %s' % (update + 1, updates, episode + 1, step + 1,
                                                           rewards[-1], mean(rewards), sum(rewards),
                                                           elapsed_time, elapsed_time / (update + 1)),
              end=end)

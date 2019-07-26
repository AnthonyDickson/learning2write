from agent import A2C, ModelI
from env import WritingEnvironment

import tensorflow as tf


# noinspection PyAbstractClass
class Model(ModelI):
    def __init__(self, n_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value_hidden = tf.keras.layers.Dense(64, activation=tf.keras.activations.elu)
        self.value_head = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)

        self.policy_hidden = tf.keras.layers.Dense(64, activation=tf.keras.activations.elu)
        self.policy_head = tf.keras.layers.Dense(n_actions, activation=tf.keras.activations.linear)

    def call(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)

        value = self.value_head(self.value_hidden(x))
        action = self.policy_head(self.policy_hidden(x))

        return action, value

    def action_value(self, state):
        return self(state)


if __name__ == '__main__':
    env = WritingEnvironment()
    model = Model(env.action_space.n)
    agent = A2C(model, env)

    agent.train(updates=10000)

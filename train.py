import plac as plac
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from env import WritingEnvironment


@plac.annotations(
    updates=plac.Annotation('How steps to train the model for.', kind='positional', type=int),
    n_workers=plac.Annotation('How many workers, or cpus, to train with')
)
def main(updates=10000, n_workers=4):
    # multiprocess environment
    env = SubprocVecEnv([lambda: WritingEnvironment() for i in range(n_workers)])

    model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="./a2c_writing_tensorboard/")
    model.learn(total_timesteps=updates)
    model.save("acktr_writing")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    plac.call(main)

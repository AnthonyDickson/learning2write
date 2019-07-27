import plac as plac
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from env import WritingEnvironment


@plac.annotations(
    updates=plac.Annotation('How steps to train the model for.', type=int, kind='option'),
    n_workers=plac.Annotation('How many workers, or cpus, to train with.', type=int, kind='option')
)
def main(updates=10000, n_workers=4):
    # multiprocess environment
    envs = SubprocVecEnv([lambda: WritingEnvironment() for _ in range(n_workers)])

    model = ACKTR(MlpPolicy, envs, verbose=1, tensorboard_log="./learning2write_tensorboard/")
    model.learn(total_timesteps=updates)
    model.save("acktr_learning2write")


if __name__ == '__main__':
    plac.call(main)

import plac as plac
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from learning2write import WritingEnvironment, get_pattern_set


@plac.annotations(
    model_path=plac.Annotation('Continue training a model specified by a path to a saved model.',
                               type=str, kind='option'),
    pattern_set=plac.Annotation('The set of patterns to use in the environment.', choices=['3x3', '5x5'],
                                kind='option', type=str),
    updates=plac.Annotation('How steps to train the model for.', type=int, kind='option'),
    n_workers=plac.Annotation('How many workers, or cpus, to train with.', type=int, kind='option')
)
def main(model_path=None, pattern_set='3x3', updates=10000, n_workers=4):
    """Train an ACKTR RL agent on the learning2write environment."""

    env = SubprocVecEnv([lambda: WritingEnvironment(get_pattern_set(pattern_set)) for _ in range(n_workers)])
    tensorboard_log = "./learning2write_%s_tensorboard/" % pattern_set

    # TODO: Make type of model configurable via cli
    if model_path:
        model = ACKTR.load(model_path, tensorboard_log=tensorboard_log)
        model.set_env(env)
    else:
        model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log)

    try:
        model.learn(total_timesteps=updates, reset_num_timesteps=model_path is None)
    except KeyboardInterrupt:
        print('Stopping training...')
    finally:
        model.save("acktr_learning2write_%s" % pattern_set)


if __name__ == '__main__':
    plac.call(main)

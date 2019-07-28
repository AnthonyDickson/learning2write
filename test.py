from datetime import datetime
from statistics import mean

import plac
from stable_baselines.common import ActorCriticRLModel

from learning2write import get_pattern_set
from learning2write.env import WritingEnvironment


@plac.annotations(
    model_path=plac.Annotation('The path and the filename of the saved model to run.',
                               type=str, kind='positional'),
    pattern_set=plac.Annotation('The set of patterns to use in the environment.', choices=['3x3', '5x5'],
                                kind='option', type=str),
    max_updates=plac.Annotation('The maximum number of steps to perform in the evironment.', type=int, kind='option'),
    max_steps=plac.Annotation('The maximum number of steps to perform per episode.', type=int, kind='option'),
    fps=plac.Annotation('How many steps to perform per second.', type=float, kind='option')
)
def main(model_path, pattern_set='3x3', max_updates=1000, max_steps=100, fps=2.0):
    """Run a model in the writing environment in test mode (i.e. no training, just predictions).

    Press `Q` or `ESCAPE` to quit at any time.
    """

    pattern_set = get_pattern_set(pattern_set)

    # TODO: Make type of model configurable via cli
    model = ActorCriticRLModel.load(model_path)

    with WritingEnvironment(pattern_set) as env:
        episode = 0
        updates = 0
        rewards = []

        while updates < max_updates:
            episode += 1
            steps, reward = run_episode(env, episode, fps, updates, max_updates, max_steps, model)
            rewards.append(reward)
            updates += steps
            print('\nEpisode %02d - Steps: %d - Episode Reward: %.2f - Smoothed Avg. Reward: %.2f'
                  % (episode, steps, reward, mean(rewards[-1 - min(len(rewards) - 1, 100):])) + ' ' * 40)

            if env.should_quit:
                break


def run_episode(env, episode, fps, updates, max_updates, max_steps, model):
    observation = env.reset()
    step = 0
    rewards = []

    for step in range(max_steps):
        start = datetime.now()
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)

        print('\rEpisode %02d - Step %02d - Reward: %.2f - Cumulative Reward: %.2f - Mean Reward: %.2f'
              % (episode, step + 1, reward, sum(rewards), mean(rewards)), end='')
        env.render()
        env.wait(1.0 / fps - (datetime.now() - start).total_seconds())  # sync steps to framerate (`fps`)
        updates += 1

        if done or env.should_quit or updates >= max_updates:
            break

    return step + 1, sum(rewards)


if __name__ == '__main__':
    plac.call(main)

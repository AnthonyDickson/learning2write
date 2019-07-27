from datetime import datetime
from statistics import mean

import plac
from stable_baselines import ACKTR

from env import WritingEnvironment


@plac.annotations(
    path_to_model=plac.Annotation('The path and the filename of the saved model to run.', type=str),
    max_updates=plac.Annotation('The maximum number of steps to perform in the evironment.', type=int, kind='option'),
    max_steps=plac.Annotation('The maximum number of steps to perform per episode.', type=int, kind='option'),
    fps=plac.Annotation('How many steps to perform per second.', type=float, kind='option')
)
def main(path_to_model='acktr_learning2write', max_updates=1000, max_steps=100, fps=2.0):
    """Run a model in the writing environment in test mode (i.e. no training, just predictions).

    Press `Q` or `ESCAPE` to quit at any time.
    """
    # TODO: Make type of model configurable via cli
    model = ACKTR.load(path_to_model)

    with WritingEnvironment() as env:
        episode = 0
        updates = 0
        rewards = []

        while updates < max_updates:
            episode += 1
            steps, reward = run_episode(env, episode, fps, updates, max_updates, max_steps, model)
            rewards.append(reward)
            updates += steps
            print('\rEpisode %02d - Steps: %d - Reward: %d - Mean Reward: %.2f'
                  % (episode, steps, reward, mean(rewards[min(len(rewards) - 1, 100):])))

            if env.should_quit:
                break


def run_episode(env, episode, fps, updates, max_updates, max_steps, model):
    observation = env.reset()
    episode_reward = 0
    step = 0

    for step in range(max_steps):
        start = datetime.now()
        action, _ = model.predict(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward

        print('\rEpisode %02d - Step %02d - Reward: %d' % (episode, step + 1, reward), end='')
        env.render()
        env.wait(1.0 / fps - (datetime.now() - start).total_seconds())  # sync steps to framerate (`fps`)
        updates += 1

        if done or env.should_quit or updates >= max_updates:
            break

    return step + 1, episode_reward


if __name__ == '__main__':
    plac.call(main)

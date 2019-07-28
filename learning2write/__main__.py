from datetime import datetime
from statistics import mean

import plac

from learning2write import WritingEnvironment, get_pattern_set, VALID_PATTERN_SETS


@plac.annotations(
    pattern_set=plac.Annotation('The set of patterns to use in the environment.', choices=VALID_PATTERN_SETS,
                                kind='option', type=str),
    max_steps=plac.Annotation('The maximum number of steps to perform per episode.', type=int, kind='option'),
    fps=plac.Annotation('How many steps to perform per second.', type=float, kind='option')
)
def main(pattern_set='3x3', max_steps=100, fps=4):
    """Run a demo of a random agent in the learning2write environment."""
    with WritingEnvironment(get_pattern_set(pattern_set)) as env:
        episode = 0
        steps = 0
        rewards = []

        while True:
            episode += 1
            episode_rewards = []
            env.reset()

            for step in range(max_steps):
                start = datetime.now()
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
                episode_rewards.append(reward)

                env.render()
                env.wait(1.0 / fps - (datetime.now() - start).total_seconds())  # sync steps to framerate (`fps`)
                print('\rEpisode %02d - Step %02d - Reward: %.2f - Cumulative Reward: %.2f - Mean Reward: %.2f'
                      % (episode, step + 1, reward, sum(episode_rewards), mean(episode_rewards)), end='')
                steps += 1

                if done or env.should_quit:
                    break

            if env.should_quit:
                break

            rewards.append(sum(episode_rewards))
            print('\rEpisode %02d - Steps: %d - Episode Reward: %.2f - Smoothed Avg. Reward: %.2f'
                  % (episode, steps, reward, mean(rewards[-1 - min(len(rewards) - 1, 100):])) + ' ' * 40)


if __name__ == '__main__':
    plac.call(main)

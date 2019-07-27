from datetime import datetime

import plac

from learning2write import WritingEnvironment, get_pattern_set


@plac.annotations(
    pattern_set=plac.Annotation('The set of patterns to use in the environment.', choices=['3x3', '5x5'],
                                kind='option', type=str),
    max_steps=plac.Annotation('The maximum number of steps to perform per episode.', type=int, kind='option'),
    fps=plac.Annotation('How many steps to perform per second.', type=float, kind='option')
)
def main(pattern_set='3x3', max_steps=100, fps=4):
    """Run a demo of a random agent in the learning2write environment."""
    with WritingEnvironment(get_pattern_set(pattern_set)) as env:
        episode = 0

        while True:
            episode += 1
            env.reset()

            for step in range(max_steps):
                start = datetime.now()
                action = env.action_space.sample()
                _, _, done, _ = env.step(action)

                env.render()
                env.wait(1.0 / fps - (datetime.now() - start).total_seconds())  # sync steps to framerate (`fps`)

                if done or env.should_quit:
                    break

            if env.should_quit:
                break


if __name__ == '__main__':
    plac.call(main)

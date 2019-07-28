"""This module defines the patterns (or symbols) to use in the learning2write environment."""

import random
from abc import ABC

import numpy as np
from mnist import MNIST

SIMPLE_PATTERN_SETS = {'3x3', '5x5'}
EMNIST_PATTERN_SETS = {'digits', 'letters', 'emnist'}
VALID_PATTERN_SETS = SIMPLE_PATTERN_SETS.union(EMNIST_PATTERN_SETS)


def get_pattern_set(pattern_set_name, batch_size=32):
    """Get an instance of a pattern set.

    :param pattern_set_name: The name of a pattern set. Valid names are those in `VALID_PATTERN_SETS`.
    :param batch_size: In the case of a MNIST based pattern set, batch size is the number of images to keep in memory.
    :return: An instance of the pattern set corresponding to the given name.
    """
    if pattern_set_name not in VALID_PATTERN_SETS:
        raise ValueError('Unrecognised pattern set \'%s\'' % pattern_set_name)

    if pattern_set_name == '3x3':
        return Patterns3x3()
    elif pattern_set_name == '5x5':
        return Patterns5x5()
    elif pattern_set_name == 'digits':
        return PatternsMNIST(pattern_set_name, batch_size)
    elif pattern_set_name == 'letters':
        return PatternsMNIST(pattern_set_name, batch_size)
    elif pattern_set_name == 'emnist':
        return PatternsMNIST('byclass', batch_size)


class PatternSet(ABC):
    """A set of patterns and symbols."""

    PATTERNS = np.array([])
    WIDTH, HEIGHT = 0, 0

    @staticmethod
    def seed(a=None):
        """Seed the random number generator used for sampling patterns.
        This behaves the same as the built-in `random`.

        :param a: The seed to use. If `None` then no seed is used..
        """
        random.seed(a)

    def sample(self) -> np.ndarray:
        """Choose a random pattern.

        :return: A randomly chosen pattern.
        """
        return random.choice(self.PATTERNS)


class Patterns3x3(PatternSet):
    """A set of 3x3 patterns, mostly consisting of letters and numbers."""

    WIDTH = HEIGHT = 3

    PATTERNS = np.array([
        # Nothing
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        # Dot
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        # Dot
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        # One
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        # Fancy one
        [[1, 1, 0],
         [0, 1, 0],
         [1, 1, 1]],
        # Four
        [[1, 0, 1],
         [1, 1, 1],
         [0, 0, 1]],
        # Seven
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
        # T for TensorFlow :)
        [[1, 1, 1],
         [0, 1, 0],
         [0, 1, 0]],
        # I/H
        [[1, 1, 1],
         [0, 1, 0],
         [1, 1, 1]],
        # X
        [[1, 0, 1],
         [0, 1, 0],
         [1, 0, 1]],
        # Y
        [[1, 0, 1],
         [0, 1, 0],
         [0, 1, 0]],
        # U/C
        [[1, 0, 1],
         [1, 0, 1],
         [1, 1, 1]],
        # O
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
    ])


class Patterns5x5(PatternSet):
    """A set of 5x5 patterns, mostly consisting of letters and numbers."""

    WIDTH = HEIGHT = 5

    PATTERNS = np.array([
        # Nothing
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Dot
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # One
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        # Fancy One
        [[0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0]],
        # Two
        [[0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # Three
        [[0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # Four
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]],
        # Five
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # Six
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # Seven
        [[0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]],
        # Eight
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # Nine
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]],
        # A
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0]],
        # B
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # C
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # D
        [[0, 1, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 0]],
        # E
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # F
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0]],
        # G
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # H
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0]],
        # I
        [[0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0]],
        # J
        [[1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 0, 1, 0, 0],
         [0, 1, 0, 0, 0]],
        # K
        [[0, 1, 0, 1, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 0, 1, 0]],
        # L
        [[0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # M
        [[1, 0, 0, 0, 1],
         [1, 1, 0, 1, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1]],
        # N
        [[1, 0, 0, 0, 1],
         [1, 1, 0, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 0, 1, 1],
         [1, 0, 0, 0, 1]],
        # O
        [[1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1]],
        # P
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0]],
        # Too hard to do a good Q :(
        # R
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 0, 1, 0]],
        # S
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # T
        [[0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        # U
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0]],
        # V
        [[1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0]],
        # W
        [[1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0]],
        # X
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1]],
        # Y
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        # Z
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 1, 1, 1, 1]],
    ])


class PatternsMNIST(PatternSet):
    WIDTH = HEIGHT = 28

    def __init__(self, dataset, batch_size=32):
        """Create a new EMNIST pattern generator.

        :param dataset: Which subset of EMNIST to use.
                        Valid choices are:
                        - byclass   : 814,255 characters. 62 unbalanced classes.
                        - bymerge   : 814,255 characters. 47 unbalanced classes.
                        - balanced  : 131,600 characters. 47 balanced classes.
                        - letters   : 145,600 characters. 26 balanced classes.
                        - digits    : 280,000 characters. 10 balanced classes.
                        - mnist     : 70,000 characters. 10 balanced classes.

        :param batch_size: The number of images to load into memory.
        """
        self.emnist = MNIST('emnist_data', mode='rounded_binarized', return_type='numpy')
        self.emnist.select_emnist(dataset)
        self.batch_size = batch_size
        self.images = self._image_gen()

    def sample(self) -> np.ndarray:
        try:
            return next(self.images)
        except StopIteration:
            # Ran out of images, start again
            self.images = self._image_gen()
            return next(self.images)

    def _image_gen(self):
        for images, _ in self.emnist.load_training_in_batches(self.batch_size):
            random.shuffle(images)
            images = images.reshape(-1, 28, 28)  # shape of EMNIST images

            for image in images:
                yield image

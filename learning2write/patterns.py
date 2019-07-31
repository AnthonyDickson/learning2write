"""This module defines the patterns (or symbols) to use in the learning2write environment."""
import abc
import random
from abc import ABC

import numpy as np
from mnist import MNIST

SIMPLE_PATTERN_SETS = {'3x3', '5x5'}
EMNIST_PATTERN_SETS = {'mnist', 'digits', 'letters', 'emnist'}
VALID_PATTERN_SETS = SIMPLE_PATTERN_SETS.union(EMNIST_PATTERN_SETS)


def get_pattern_set(pattern_set_name, rotate_patterns=False, batch_size=32):
    """Get an instance of a pattern set.

    :param pattern_set_name: The name of a pattern set. Valid names are those in `VALID_PATTERN_SETS`.
    :param rotate_patterns: Whether or not patterns returned by `sample()` should be randomly rotated.
    :param batch_size: In the case of a MNIST based pattern set, batch size is the number of images to keep in memory.
    :return: An instance of the pattern set corresponding to the given name.
    """
    if pattern_set_name not in VALID_PATTERN_SETS:
        raise ValueError('Unrecognised pattern set \'%s\'' % pattern_set_name)

    if pattern_set_name == '3x3':
        return Patterns3x3(rotate_patterns)
    elif pattern_set_name == '5x5':
        return Patterns5x5(rotate_patterns)
    elif pattern_set_name in EMNIST_PATTERN_SETS:
        if pattern_set_name == 'emnist':
            return PatternsMNIST('byclass', batch_size, rotate_patterns)
        else:
            return PatternsMNIST(pattern_set_name, batch_size, rotate_patterns)


class PatternSet(ABC):
    """A set of patterns and symbols."""

    patterns = np.array([])
    width, height = 0, 0

    def __init__(self, rotate_patterns=False):
        """Create a new pattern set.

        :param rotate_patterns: Whether or not patterns returned by `sample()` should be randomly rotated.
        """
        self.rotate_patterns = rotate_patterns

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def __getitem__(self, item):
        return self.patterns[item]

    @staticmethod
    def seed(a=None):
        """Seed the random number generator used for sampling patterns.
        This behaves the same as the built-in `random`.

        :param a: The seed to use. If `None` then no seed is used.
        """
        random.seed(a)

    def sample(self) -> np.ndarray:
        """Choose a random pattern.

        :return: A randomly chosen pattern.
        """
        pattern = random.choice(self.patterns)
        return np.rot90(pattern, k=random.randint(0, 3)) if self.rotate_patterns else pattern


class Patterns3x3(PatternSet):
    """A set of 3x3 patterns, mostly consisting of letters and numbers."""

    width = height = 3

    patterns = np.array([
        # Nothing
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        # All Blacks
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        # Dot
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]],
        # Diag
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        # Lines
        [[1, 1, 1],
         [0, 0, 0],
         [1, 1, 1]],
        # Corners
        [[1, 0, 1],
         [0, 0, 0],
         [1, 0, 1]],
        # Cross
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        # Middle Edges
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]],
        # Pyramid
        [[0, 0, 0],
         [0, 1, 0],
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
        # V
        [[1, 0, 1],
         [1, 0, 1],
         [0, 1, 0]],
        # O
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
    ])

    @property
    def name(self) -> str:
        return '3x3'


class Patterns5x5(PatternSet):
    """A set of 5x5 patterns, mostly consisting of letters and numbers."""

    width = height = 5

    patterns = np.array([
        # Nothing
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Everything
        [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]],
        # Dot
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # Line with a Dot
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]],
        # Line
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        # Line in the middle
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        # Thicc line
        [[0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 1, 1, 0]],
        # Two Lines
        [[1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1]],
        # Three Lines
        [[1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1]],
        # Diagonal
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]],
        # Two Diagonals
        [[1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]],
        # Three Diagonals
        [[1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 1]],
        # Corners
        [[1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1]],
        # Bigger corners
        [[1, 1, 0, 1, 1],
         [1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [1, 1, 0, 1, 1]],
        # Equals sign
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]],
        # Not an Equals sign
        [[0, 0, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 0, 0, 0]],
        # Less than
        [[0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0]],
        # Less than or equals
        [[0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # Greater than
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0]],
        # Greater than or equals
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # Hourglass
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0]],
        # Checkerboard
        [[1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1]],
        # #hashtag
        [[0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 0, 1, 0]],
        # Arrow
        [[0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [1, 0, 1, 0, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        # Diagonal Arrow
        [[0, 1, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 0, 1],
         [0, 1, 0, 0, 1],
         [1, 0, 0, 0, 0]],
        # Diamond
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0]],
        # Smiley
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # Not Smiley
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]],
        # Definitely Not Smiley
        [[0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1]],
        # Can't believe it's not smiley
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1]],
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
        # e
        [[1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1]],
        # F
        [[0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0]],
        # f
        [[0, 0, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
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
        # j
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
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
        # u
        [[1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1]],
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
        # Zea
        [[0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0]],
        # Zed
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 1, 1, 1, 0]],
        # A cross zed
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 0, 0, 0],
         [1, 1, 1, 1, 0]],
    ])

    @property
    def name(self) -> str:
        return '5x5'


class PatternsMNIST(PatternSet):
    width = height = 28

    def __init__(self, dataset, batch_size=32, rotate_patterns=False):
        """Create a new EMNIST pattern set.

        :param rotate_patterns: Whether or not patterns returned by `sample()` should be randomly rotated.
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
        super().__init__(rotate_patterns)

        self.emnist = MNIST('emnist_data', mode='rounded_binarized', return_type='numpy')
        self.emnist.select_emnist(dataset)
        self.batch_size = batch_size
        self.images = self._image_gen()
        self._name = 'emnist' if dataset in {'byclass', 'bymerge', 'balanced'} else dataset

    @property
    def name(self) -> str:
        return self._name

    def sample(self) -> np.ndarray:
        try:
            image = next(self.images)
        except StopIteration:
            # Ran out of images, start again
            self.images = self._image_gen()
            image = next(self.images)

        return np.rot90(image, k=random.randint(0, 3)) if self.rotate_patterns else image

    def _image_gen(self):
        for images, _ in self.emnist.load_training_in_batches(self.batch_size):
            random.shuffle(images)
            self.patterns = images.reshape(-1, self.width, self.height)

            for image in self.patterns:
                yield image

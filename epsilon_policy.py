from __future__ import absolute_import, division, print_function

import abc
from six import with_metaclass


__all__ = ['EpsilonPolicy', 'LinearlyDecayingEpsilonPolicy', 'ExponentiallyDecayingEpsilonPolicy']


class EpsilonPolicy(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def get_epsilon(self, current_epsilon, triggered=False):
        raise NotImplemented


class LinearlyDecayingEpsilonPolicy(EpsilonPolicy):
    def __init__(self, start_step, start_value, difference, lower_bound=0.0):
        self.start_step = start_step
        self.start_value = start_value
        self.difference = difference
        self.lower_bound = lower_bound
        self._current_step = 0

    def get_epsilon(self, current_epsilon, triggered=False):
        self._current_step += 1
        if self._current_step > self.start_step:
            value = current_epsilon - self.difference
        else:
            value = self.start_value
        return max(value, self.lower_bound)


class ExponentiallyDecayingEpsilonPolicy(EpsilonPolicy):
    def __init__(self, start_step, start_value, factor, lower_bound=0.0):
        self.start_step = start_step
        self.start_value = start_value
        self.factor = factor
        self.lower_bound = lower_bound
        self._current_step = 0

    def get_epsilon(self, current_epsilon, triggered=False):
        self._current_step += 1
        if self._current_step > self.start_step:
            value = current_epsilon * self.factor
        else:
            value = self.start_value
        return max(value, self.lower_bound)

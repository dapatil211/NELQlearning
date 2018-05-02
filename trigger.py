from __future__ import absolute_import, division, print_function

import abc
from six import with_metaclass

__all__ = ['Trigger', 'NoTrigger', 'MACTrigger']


class Trigger(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def should_trigger(self, reward, loss):
        raise NotImplemented


class NoTrigger(Trigger):
    def should_trigger(self, reward, loss):
        return False


class MACTrigger(Trigger):
    def __init__(self, slow_factor=0.00001, fast_factor=0.001):
        self.slow_factor = slow_factor
        self.fast_factor = fast_factor
        self._slow_ma = None
        self._fast_ma = None
        self._prev_slow = False

    def should_trigger(self, reward, loss):
        if self._slow_ma is None:
            self._slow_ma = reward
            self._fast_ma = reward
            self._prev_slow = False
        self._slow_ma = self._update_ma(self._slow_ma, reward, self.slow_factor)
        self._fast_ma = self._update_ma(self._fast_ma, reward, self.fast_factor)
        if self._prev_slow and self._slow_ma < self._fast_ma:
            self._prev_slow = False
            return True
        if not self._prev_slow and self._slow_ma > self._fast_ma:
            self._prev_slow = True
        return False

    @staticmethod
    def _update_ma(ma, value, factor):
        return factor * value + (1 - factor) * ma

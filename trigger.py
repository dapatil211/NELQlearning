from __future__ import absolute_import, division, print_function

import abc
import numpy as np
from collections import deque
from six import with_metaclass

__all__ = ['Trigger', 'NoTrigger', 'MACTrigger', 'LTATrigger', 'LTAMACTrigger']


class Trigger(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def should_trigger(self, reward, loss):
        raise NotImplemented


class NoTrigger(Trigger):
    def should_trigger(self, reward, loss):
        return False


class MACTrigger(Trigger):
    def __init__(self, slow_factor=0.0001, fast_factor=0.01, keep_triggering=False):
        self.slow_factor = slow_factor
        self.fast_factor = fast_factor
        self.keep_triggering = keep_triggering
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
        if self._slow_ma < self._fast_ma and (self._prev_slow or self.keep_triggering):
            self._prev_slow = False
            return True
        if self._prev_slow and self._slow_ma > self._fast_ma:
            self._prev_slow = True
        return False

    @staticmethod
    def _update_ma(ma, value, factor):
        return factor * value + (1 - factor) * ma


class LTATrigger(Trigger):
    def __init__(self, min_steps_between_triggers=5000, window=1000):
        self.min_steps_between_triggers = min_steps_between_triggers
        self.steps_since_trigger = 0
        self.steps = 0
        self.window = window
        self.reward_window = deque(maxlen=window)
        self.loss_window = deque(maxlen=window)
        self.reward_average = 0.
    
    def should_trigger(self, reward, loss):
        self.reward_window.append(reward)
        self.loss_window.append(loss)
        current_reward_avg = np.mean(self.reward_window)
        self.reward_average = (self.reward_average * self.steps + reward) / (self.steps + 1.)
        self.steps += 1

        if current_reward_avg < self.reward_average:
            losses = np.array(self.loss_window)
            loss_avg_1 = np.mean(losses[:len(losses)//2])
            loss_avg_2 = np.mean(losses[len(losses)//2:])
            if loss_avg_1 < loss_avg_2:
                if self.steps_since_trigger >= self.min_steps_between_triggers:
                    self.steps_since_trigger = 0
                    return True
        self.steps_since_trigger += 1
        return False


class LTAMACTrigger(Trigger):
    def __init__(self, slow_factor=0.0001, fast_factor=0.01, min_steps_between_triggers=5000):
        self.slow_factor = slow_factor
        self.fast_factor = fast_factor
        self._slow_ma_r = None
        self._fast_ma_r = None
        self._slow_ma_l = None
        self._fast_ma_l = None
        self.min_steps_between_triggers = min_steps_between_triggers
        self.steps_since_trigger = 0

    def should_trigger(self, reward, loss):
        if self._slow_ma_r is None:
            self._slow_ma_r = reward
            self._fast_ma_r = reward
            self._slow_ma_l = loss
            self._fast_ma_l = loss
        self._slow_ma_r = self._update_ma(self._slow_ma_r, reward, self.slow_factor)
        self._fast_ma_r = self._update_ma(self._fast_ma_r, reward, self.fast_factor)
        self._slow_ma_l = self._update_ma(self._slow_ma_l, loss, self.slow_factor)
        self._fast_ma_l = self._update_ma(self._fast_ma_l, loss, self.fast_factor)
        if self._slow_ma_r > self._fast_ma_r and self._slow_ma_l < self._fast_ma_l:
            if self.steps_since_trigger >= self.min_steps_between_triggers:
                self.steps_since_trigger = 0
                return True
        self.steps_since_trigger += 1
        return False

    @staticmethod
    def _update_ma(ma, value, factor):
        return factor * value + (1 - factor) * ma

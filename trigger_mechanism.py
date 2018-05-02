from __future__ import absolute_import, division, print_function

import abc
from six import with_metaclass

__all__ = ['TriggerMechanism', 'NoTriggerMechanism']


class TriggerMechanism(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def should_trigger(self, reward, loss):
        raise NotImplemented


class NoTriggerMechanism(TriggerMechanism):
    def should_trigger(self, reward, loss):
        return False

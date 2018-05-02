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


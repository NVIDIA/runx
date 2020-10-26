import copy
import itertools
import math
import random
from typing import List, Union, Any, Iterable, Dict


Primitive = Union[int, float, str]


class NotSupportedException(Exception):
    pass


class BaseDistribution:
    def sample(self) -> Primitive:
        """Returns a single sample from the distribution."""
        raise ValueError("Not implemented!")

    @property
    def is_discrete(self) -> bool:
        """Returns whether the set of possible values can be enumerated."""
        return False

    def enumerate(self) -> Iterable[Primitive]:
        raise NotSupportedException(f'The distribution "{type(self)}" is not discrete!')


class CategoricalDistribution(BaseDistribution):
    def __init__(self, categories: List[Primitive]):
        self.categories = categories

    def sample(self):
        return random.choice(self.categories)

    @property
    def is_discrete(self):
        return True

    def enumerate(self):
        return self.categories


class UniformDistribution(BaseDistribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self):
        return random.uniform(self.low, self.high)


class NormalDistribution(BaseDistribution):
    def __init__(self, mean: float=0, std: float=1):
        self.mean = mean
        self.std = std

    def sample(self):
        return random.normalvariate(self.mean, self.std)


class LogUniformDistribution(BaseDistribution):
    def __init__(self, low: float, high: float, base: float=math.e):
        self.low = low
        self.high = high
        self.base = base

    def sample(self):
        log_low = math.log(self.low, self.base)
        log_high = math.log(self.high, self.base)

        v = random.uniform(log_low, log_high)

        if self.base is None:
            return math.exp(v)
        else:
            return self.base ** v


def convert_to_distribution(val: Union[Primitive, List[Primitive], BaseDistribution]) -> BaseDistribution:
    if isinstance(val, BaseDistribution):
        return val
    if isinstance(val, (list, tuple)):
        return CategoricalDistribution(val)

    return CategoricalDistribution([val])


def can_enumerate(distributions: List[BaseDistribution]) -> bool:
    return all(d.is_discrete for d in distributions)


def enumerate_dists(distributions: List[BaseDistribution]) -> List[List[Primitive]]:
    all_prims = [list(d.enumerate()) for d in distributions]

    expanded_prims = itertools.product(*all_prims)

    realizations = list(expanded_prims)

    return realizations


def sample_dists(distributions: List[BaseDistribution]) -> List[Primitive]:
    return [d.sample() for d in distributions]

_FACTORIES = {
    'uniform': UniformDistribution,
    'normal': NormalDistribution,
    'log_uniform': LogUniformDistribution,
}

def load_config(cfg: Union[Any, Dict[str, Primitive]]):
    if isinstance(cfg, dict) and 'distribution' in cfg:
        cfg = copy.copy(cfg)
        dist_name = cfg['distribution']
        del cfg['distribution']

        v = _FACTORIES[dist_name](**cfg)
    else:
        v = cfg

    return convert_to_distribution(v)
import copy
import itertools
import math
import random
from typing import List, Union, Any, Iterable, Dict, Tuple


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
    def __init__(self, categories: List[Primitive], literal=False):
        self.categories = categories
        self.literal = literal

    def sample(self):
        return random.choice(self.categories)

    @property
    def is_discrete(self):
        return self.literal

    def enumerate(self):
        return self.categories


class MultinomialDistribution(CategoricalDistribution):
    def __init__(self, categories: List[Primitive], weights: List[float]):
        super().__init__(categories)

        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

    def sample(self):
        return random.choices(self.categories, self.weights, k=1)[0]


class UniformDistribution(BaseDistribution):
    def __init__(self, low: float, high: float, is_integer: bool=False):
        self.low = low
        self.high = high
        self.is_integer = is_integer

    def sample(self):
        val = random.uniform(self.low, self.high)
        if self.is_integer:
            val = int(round(val))
        return val


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
        return CategoricalDistribution(val, literal=True)

    return CategoricalDistribution([val], literal=True)


def discrete_continuous_split(distributions: List[BaseDistribution]) \
        -> Tuple[List[Tuple[int, BaseDistribution]], List[Tuple[int, BaseDistribution]]]:
    disc = []
    cont = []
    for i, dist in enumerate(distributions):
        if dist.is_discrete:
            disc.append((i, dist))
        else:
            cont.append((i, dist))
    return disc, cont


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
    'categorical': CategoricalDistribution,
    'multinomial': MultinomialDistribution,
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


def enumerate_hparams(hparams, num_trials) -> List[List[Primitive]]:
    for k in hparams:
        hparams[k] = load_config(hparams[k])

    discrete_dists, continuous_dists = discrete_continuous_split(hparams.values())

    if discrete_dists and not continuous_dists:
        realizations = enumerate_dists([d[1] for d in discrete_dists])

        if num_trials == 0 or num_trials > len(realizations):
            return realizations
        else:
            return random.choices(realizations, k=num_trials)

    if num_trials == 0:
        raise ValueError("The number of trials must be specified"
                            " when optimizing over continuous"
                            " distributions")

    if continuous_dists and not discrete_dists:
        continuous_dists = [d[1] for d in continuous_dists]
        realizations = [
            sample_dists(continuous_dists)
            for _ in range(num_trials)
        ]

        return realizations

    discrete_realizations = enumerate_dists([d[1] for d in discrete_dists])

    required_trials = math.ceil(num_trials / len(discrete_realizations)) * len(discrete_realizations)

    if required_trials > 2 * num_trials:
        raise ValueError(f"The number of required trials - {required_trials} - is more than"
                         f" double the number of allotted trials - {num_trials}"
                         f" in order to satisfy the grid and sampling constraints."
                         f" Please increase the number of trials, or explicitly define"
                         f" one or more of the discrete sets using the 'categorical'"
                         f" distribution.")
    elif required_trials > num_trials:
        print(f'Warning: Requiring {required_trials} total trials instead of {num_trials} to satisfy constraints.')

    num_trials = required_trials
    num_trials_per_disc = num_trials // len(discrete_realizations)

    cont_dists = [d[1] for d in continuous_dists]

    realizations = []
    for disc_realization in discrete_realizations:
        for _ in range(num_trials_per_disc):
            ret = [None for _ in range(len(hparams))]
            for k, r in enumerate(disc_realization):
                ret[discrete_dists[k][0]] = r

            cont_realization = sample_dists(cont_dists)
            for k, r in enumerate(cont_realization):
                ret[continuous_dists[k][0]] = r

            realizations.append(ret)

    return realizations
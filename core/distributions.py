from typing import Any, List, Optional, Sequence
import numpy as np


def sample_uniform_int(
    rng: np.random.Generator, low: int, high: int, size: int
) -> np.ndarray:
    """Sample integers uniformly in [low, high]."""
    return rng.integers(low, high + 1, size=size)


def sample_truncated_normal_int(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: int,
    high: int,
    size: int,
) -> np.ndarray:
    """Sample integer values from a normal distribution clipped to [low, high]."""
    raw = rng.normal(loc=mean, scale=std, size=size)
    clipped = np.clip(raw, low, high)
    return np.round(clipped).astype(int)


def sample_truncated_normal_float(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    decimals: int,
    size: int,
) -> np.ndarray:
    """Sample continuous float values from a normal distribution clipped to [low, high]."""
    raw = rng.normal(loc=mean, scale=std, size=size)
    clipped = np.clip(raw, low, high)
    return np.round(clipped, decimals=decimals)


def sample_weighted_categorical(
    rng: np.random.Generator,
    categories: Sequence[Any],
    weights: Sequence[float],
    size: int,
) -> List[Any]:
    """Sample categorical values given probabilities/weights."""
    norm_weights = np.array(weights, dtype=float)
    norm_weights = norm_weights / norm_weights.sum()
    indices = rng.choice(len(categories), size=size, p=norm_weights)
    return [categories[i] for i in indices]


def sample_bernoulli(
    rng: np.random.Generator, p: float, size: int
) -> np.ndarray:
    """Sample boolean values with probability p of True (or 1)."""
    return rng.random(size=size) < p


def sample_vocabulary(
    rng: np.random.Generator, vocabulary: Sequence[str], size: int
) -> List[str]:
    """Sample words or names uniformly from a vocabulary list."""
    indices = rng.integers(0, len(vocabulary), size=size)
    return [vocabulary[i] for i in indices]


def generate_unique_ids(prefix: str, count: int, digits: int = 6) -> List[str]:
    """Generate deterministic unique IDs: e.g. CUST-000001, CUST-000002."""
    fmt = f"{prefix}-{{:0{digits}d}}"
    return [fmt.format(i + 1) for i in range(count)]

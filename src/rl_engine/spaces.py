import numpy as np
from typing import Sequence, Optional, Union, Any


class Space:
    """强化学习空间基类 (兼容 Gymnasium 接口规范)"""
    def __init__(self, shape: Optional[Sequence[int]] = None, dtype: Optional[np.dtype] = None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = np.dtype(dtype) if dtype is not None else None

    def sample(self) -> np.ndarray:
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        raise NotImplementedError


class Box(Space):
    """连续有界空间 (Continuous Box Space)"""
    def __init__(
        self,
        low: Union[float, int, np.ndarray],
        high: Union[float, int, np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: np.dtype = np.float32,
    ):
        dtype = np.dtype(dtype)
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if shape is None:
                shape = (1,)
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        else:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            if shape is None:
                shape = self.low.shape

        super().__init__(shape=shape, dtype=dtype)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return bool(
            arr.shape == self.shape
            and np.all(arr >= self.low - 1e-6)
            and np.all(arr <= self.high + 1e-6)
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low.min()}, high={self.high.max()}, shape={self.shape}, dtype={self.dtype})"


class Discrete(Space):
    """离散空间 (Discrete Space)"""
    def __init__(self, n: int, start: int = 0):
        assert n > 0, "n must be positive"
        self.n = n
        self.start = start
        super().__init__(shape=(), dtype=np.int64)

    def sample(self) -> int:
        return int(np.random.randint(self.start, self.start + self.n))

    def contains(self, x: int) -> bool:
        if not isinstance(x, (int, np.integer)):
            return False
        return bool(self.start <= x < self.start + self.n)

    def __repr__(self) -> str:
        return f"Discrete(n={self.n}, start={self.start})"

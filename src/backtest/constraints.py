"""组合优化约束条件

定义高级量化组合优化的各种约束，可组合使用：
- PositionLimit: 单票最大权重
- IndustryLimit: 行业敞口上限
- TurnoverLimit: 换手率约束
- LongOnly: 仅做多
- FullInvestment: 满仓投资
- TrackingErrorLimit: 跟踪误差上限
- StyleExposureLimit: Barra风格因子暴露限制
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class ConstraintBase(ABC):
    """约束基类"""

    name: str = "base"

    @abstractmethod
    def to_scipy(self, n: int) -> dict:
        """转换为scipy.optimize.minimize约束"""
        pass

    @abstractmethod
    def check(self, weights: np.ndarray) -> tuple:
        """检查约束是否满足"""
        pass


class PositionLimit(ConstraintBase):
    """单票权重上限"""

    name = "position_limit"

    def __init__(self, max_weight: float = 0.05):
        self.max_weight = max_weight

    def to_scipy(self, n: int):
        def con(w):
            return self.max_weight - w
        return {"type": "ineq", "fun": con}

    def check(self, weights):
        violations = weights[weights > self.max_weight + 1e-6]
        if len(violations) > 0:
            return False, f"单票超限: max={violations.max():.4f} > {self.max_weight}"
        return True, "OK"


class IndustryLimit(ConstraintBase):
    """行业敞口约束"""

    name = "industry_limit"

    def __init__(self, max_exposure: float = 0.30,
                 industry_map: Optional[pd.Series] = None):
        self.max_exposure = max_exposure
        self.industry_map = industry_map

    def to_scipy(self, n: int):
        if self.industry_map is None:
            return None
        constraints = []
        for ind_name in self.industry_map.unique():
            mask = (self.industry_map.values == ind_name)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            def con(w, idx=idx):
                return self.max_exposure - w[idx].sum()
            constraints.append({"type": "ineq", "fun": con})
        return constraints

    def check(self, weights):
        if self.industry_map is None:
            return True, "OK (无行业映射)"
        w_series = pd.Series(weights, index=self.industry_map.index)
        by_industry = w_series.groupby(self.industry_map).sum()
        violations = by_industry[by_industry > self.max_exposure + 1e-6]
        if len(violations) > 0:
            return False, f"行业超限: {violations.to_dict()}"
        return True, "OK"


class TurnoverLimit(ConstraintBase):
    """换手率约束"""

    name = "turnover_limit"

    def __init__(self, max_turnover: float = 0.50,
                 prev_weights: Optional[np.ndarray] = None):
        self.max_turnover = max_turnover
        self.prev_weights = prev_weights

    def to_scipy(self, n: int):
        if self.prev_weights is None:
            return None
        def con(w):
            return self.max_turnover - np.abs(w - self.prev_weights).sum()
        return {"type": "ineq", "fun": con}

    def check(self, weights):
        if self.prev_weights is None:
            return True, "OK (无上一期权重)"
        turnover = np.abs(weights - self.prev_weights).sum()
        if turnover > self.max_turnover + 1e-6:
            return False, f"换手超限: {turnover:.4f} > {self.max_turnover}"
        return True, "OK"


class LongOnly(ConstraintBase):
    """仅做多约束"""

    name = "long_only"

    def to_scspy(self, n: int):
        return {"type": "ineq", "fun": lambda w: w}

    def to_scipy(self, n: int):
        return {"type": "ineq", "fun": lambda w: w}

    def check(self, weights):
        neg = weights[weights < -1e-6]
        if len(neg) > 0:
            return False, f"存在负权重: min={neg.min():.4f}"
        return True, "OK"


class FullInvestment(ConstraintBase):
    """满仓约束：权重之和 = 1"""

    name = "full_investment"

    def to_scipy(self, n: int):
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def check(self, weights):
        total = weights.sum()
        if abs(total - 1.0) > 1e-4:
            return False, f"非满仓: sum={total:.4f}"
        return True, "OK"


class TrackingErrorLimit(ConstraintBase):
    """跟踪误差约束"""

    name = "tracking_error_limit"

    def __init__(self, max_te: float = 0.05,
                 benchmark_weights: Optional[np.ndarray] = None,
                 cov_matrix: Optional[np.ndarray] = None):
        self.max_te = max_te
        self.benchmark_weights = benchmark_weights
        self.cov_matrix = cov_matrix

    def to_scipy(self, n: int):
        if self.benchmark_weights is None or self.cov_matrix is None:
            return None
        def con(w):
            diff = w - self.benchmark_weights
            te = np.sqrt(diff @ self.cov_matrix @ diff)
            return self.max_te - te
        return {"type": "ineq", "fun": con}

    def check(self, weights):
        if self.benchmark_weights is None or self.cov_matrix is None:
            return True, "OK (无基准/协方差)"
        diff = weights - self.benchmark_weights
        te = np.sqrt(diff @ self.cov_matrix @ diff)
        if te > self.max_te + 1e-6:
            return False, f"跟踪误差超限: {te:.4f} > {self.max_te}"
        return True, "OK"


@dataclass
class PortfolioConstraints:
    """约束集合：管理多个约束条件"""
    constraints: list = field(default_factory=list)

    def add(self, constraint: ConstraintBase):
        self.constraints.append(constraint)
        return self

    def to_scipy_list(self, n: int) -> list:
        """转换为scipy约束列表"""
        result = []
        for c in self.constraints:
            sc = c.to_scipy(n)
            if sc is None:
                continue
            if isinstance(sc, list):
                result.extend(sc)
            else:
                result.append(sc)
        return result

    def check_all(self, weights: np.ndarray) -> dict:
        """检查所有约束"""
        results = {}
        all_pass = True
        for c in self.constraints:
            passed, msg = c.check(weights)
            results[c.name] = {"pass": passed, "message": msg}
            if not passed:
                all_pass = False
        results["_all_pass"] = all_pass
        return results


def default_constraints(max_position: float = 0.05,
                        max_industry: float = 0.30,
                        industry_map: Optional[pd.Series] = None,
                        prev_weights: Optional[np.ndarray] = None) -> PortfolioConstraints:
    """创建默认约束集"""
    pc = PortfolioConstraints()
    pc.add(LongOnly())
    pc.add(FullInvestment())
    pc.add(PositionLimit(max_position))
    if industry_map is not None:
        pc.add(IndustryLimit(max_industry, industry_map))
    if prev_weights is not None:
        pc.add(TurnoverLimit(0.50, prev_weights))
    return pc

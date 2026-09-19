from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from src.rl_engine.spaces import Box, Discrete


class PortfolioTradingEnv:
    """
    符合 Gymnasium 规范的投资组合强化学习交易环境 (Portfolio Trading Environment)。
    
    强化学习 MDP 框架：
    - Observation (状态空间):
      包含最近 lookback_window 个交易日的资产时序收益率特征 + 当前投资组合在各资产上的实际持仓权重及现金比例。
    - Action (动作空间):
      各资产的目标连续配置权重向量 a in [0, 1]^N，经风控约束投影满足单票上限与现金缓冲区。
    - Reward (即时奖励):
      组合单步净收益 (扣除换手手续费与冲击成本) 减去下行波动风险惩罚项。
    """

    def __init__(
        self,
        returns_data: Union[np.ndarray, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        lookback_window: int = 20,
        initial_cash: float = 1_000_000.0,
        cost_rate: float = 0.001,
        risk_penalty_coeff: float = 0.5,
        max_stock_weight: float = 0.20,
        min_cash_ratio: float = 0.05,
    ):
        if isinstance(returns_data, pd.DataFrame):
            self.symbols = symbols or list(returns_data.columns)
            self.returns_matrix = returns_data.values.astype(np.float32)
        else:
            self.returns_matrix = np.asarray(returns_data, dtype=np.float32)
            self.symbols = symbols or [f"asset_{i}" for i in range(self.returns_matrix.shape[1])]

        self.num_steps, self.num_assets = self.returns_matrix.shape
        assert self.num_steps > lookback_window + 2, "returns_data length must be > lookback_window + 2"
        assert self.num_assets > 0, "num_assets must be > 0"

        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.cost_rate = cost_rate
        self.risk_penalty_coeff = risk_penalty_coeff
        self.max_stock_weight = max_stock_weight
        self.min_cash_ratio = min_cash_ratio

        # 状态空间维度：(lookback_window * num_assets) + num_assets (当前权重) + 1 (当前现金比率)
        self.obs_dim = (self.lookback_window * self.num_assets) + self.num_assets + 1
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 动作空间：各资产的目标连续权重
        self.action_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

        # 内部动态状态
        self.current_step = 0
        self.current_equity = self.initial_cash
        self.current_weights = np.zeros(self.num_assets, dtype=np.float32)
        self.current_cash_ratio = 1.0
        self.history_nav: List[float] = []
        self.history_returns: List[float] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境至初始状态"""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.lookback_window
        self.current_equity = self.initial_cash
        self.current_weights = np.zeros(self.num_assets, dtype=np.float32)
        self.current_cash_ratio = 1.0
        self.history_nav = [self.initial_cash]
        self.history_returns = []

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "equity": self.current_equity,
            "cash_ratio": self.current_cash_ratio,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行单步动作转移：
        1. 动作映射与风控合规约束截断 (单票上限 & 现金缓冲)；
        2. 计算换手冲击成本并扣除；
        3. 根据当期资产真实收益率结转账户净值；
        4. 计算风险调整回报 (Reward)；
        5. 步进并返回 (obs, reward, terminated, truncated, info)。
        """
        raw_action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        target_weights = self._project_weights(raw_action)

        # 1. 计算换手率与交易成本
        turnover = float(np.sum(np.abs(target_weights - self.current_weights)))
        cost = turnover * self.cost_rate

        # 2. 当期资产收益率结算
        asset_returns = self.returns_matrix[self.current_step]
        gross_return = float(np.sum(target_weights * asset_returns))
        net_return = gross_return - cost

        # 3. 净值结转
        self.current_equity = max(0.0, self.current_equity * (1.0 + net_return))
        self.history_nav.append(self.current_equity)
        self.history_returns.append(net_return)

        # 4. 下一步持仓状态自然漂移 (标的涨跌导致被动权重变动)
        if 1.0 + gross_return > 1e-6:
            drifted_weights = target_weights * (1.0 + asset_returns) / (1.0 + gross_return)
        else:
            drifted_weights = target_weights
        self.current_weights = drifted_weights
        self.current_cash_ratio = max(0.0, 1.0 - float(np.sum(self.current_weights)))

        # 5. 奖励函数设计 (收益激励 - 换手惩罚 - 下行负收益方差惩罚)
        downside_penalty = (net_return ** 2) if net_return < 0.0 else 0.0
        reward = net_return - (self.risk_penalty_coeff * downside_penalty)

        # 6. 推进步长与终止判定
        self.current_step += 1
        terminated = bool(
            self.current_step >= self.num_steps - 1
            or self.current_equity < 0.2 * self.initial_cash  # 80% 破产止损线
        )
        truncated = False

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "equity": self.current_equity,
            "net_return": net_return,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "target_weights": {sym: round(float(w), 4) for sym, w in zip(self.symbols, target_weights)},
        }

        return obs, float(reward), terminated, truncated, info

    def _project_weights(self, raw_action: np.ndarray) -> np.ndarray:
        """风控约束投影：满足单票持仓上限与预留最低现金缓冲"""
        sum_act = np.sum(raw_action)
        if sum_act <= 1e-6:
            return np.zeros(self.num_assets, dtype=np.float32)

        # 允许投向股票的总权重上限
        max_investable = max(0.0, 1.0 - self.min_cash_ratio)
        norm_weights = (raw_action / sum_act) * max_investable

        # 施加单票持仓上限 max_stock_weight 截断
        clipped_weights = np.minimum(norm_weights, self.max_stock_weight)
        
        # 再次归一化校验
        if np.sum(clipped_weights) > max_investable:
            clipped_weights = (clipped_weights / np.sum(clipped_weights)) * max_investable

        return clipped_weights.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """抽取并拼接当前状态特征向量"""
        # 截取 [t - lookback_window : t] 的行情收益率窗口
        start = self.current_step - self.lookback_window
        end = self.current_step
        window_returns = self.returns_matrix[start:end].flatten()

        # 拼接：窗口行情特征 + 当前权重 + 现金比例
        obs = np.concatenate([
            window_returns,
            self.current_weights,
            np.array([self.current_cash_ratio], dtype=np.float32)
        ]).astype(np.float32)
        return obs

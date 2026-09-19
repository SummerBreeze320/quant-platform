from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCriticNetwork(nn.Module):
    """
    连续动作空间的高斯 Actor-Critic 策略与价值联合深度神经网络。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 共享特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor 策略输出头：均值与可学习的 log_std
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # 将资产权重意图映射在 (0, 1) 范围
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

        # Critic 状态价值输出头
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.feature_net(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 1.0))
        dist = Normal(action_mean, action_std)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOAgent:
    """
    基于 Proximal Policy Optimization (PPO-Clip) 的量化投资组合资产配置智能体。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff

        # 采样缓存轨迹
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[np.ndarray] = []
        self.reward_buffer: List[float] = []
        self.value_buffer: List[float] = []
        self.log_prob_buffer: List[float] = []
        self.done_buffer: List[bool] = []

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """根据当前状态输出配置动作"""
        self.network.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t, log_prob_t, value_t = self.network.get_action(obs_t, deterministic=deterministic)

        action = action_t.squeeze(0).cpu().numpy()
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        return action, log_prob, value

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_prob_buffer.append(log_prob)
        self.done_buffer.append(done)

    def compute_gae(self, last_value: float, last_done: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计 (Generalized Advantage Estimation)"""
        rewards = self.reward_buffer
        values = self.value_buffer + [last_value]
        dones = self.done_buffer + [last_done]

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae

        returns = advantages + np.array(values[:n], dtype=np.float32)

        # 优势标准化
        adv_std = np.std(advantages)
        if adv_std > 1e-6:
            advantages = (advantages - np.mean(advantages)) / adv_std

        return (
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            torch.as_tensor(returns, dtype=torch.float32, device=self.device),
        )

    def update(
        self,
        last_value: float = 0.0,
        last_done: bool = True,
        epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """执行 PPO 策略更新迭代"""
        if len(self.obs_buffer) == 0:
            return {}

        self.network.train()
        advantages, returns = self.compute_gae(last_value, last_done)

        obs_tensor = torch.as_tensor(np.array(self.obs_buffer), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(np.array(self.action_buffer), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.array(self.log_prob_buffer), dtype=torch.float32, device=self.device)

        dataset_size = len(self.obs_buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates_count = 0

        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]

                b_obs = obs_tensor[batch_idx]
                b_acts = actions_tensor[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                new_log_probs, entropy, new_values = self.network.evaluate_actions(b_obs, b_acts)

                # Ratio: r(theta) = exp(new_lp - old_lp)
                ratios = torch.exp(new_log_probs - b_old_lp)

                # Clipped surrogate objective
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, b_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + (self.value_loss_coeff * value_loss) + (self.entropy_coeff * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                updates_count += 1

        self.reset_buffer()
        return {
            "policy_loss": round(total_policy_loss / max(1, updates_count), 6),
            "value_loss": round(total_value_loss / max(1, updates_count), 6),
            "entropy": round(total_entropy / max(1, updates_count), 6),
        }

    def predict_target_weights(
        self,
        obs: np.ndarray,
        symbols: List[str],
        max_stock_weight: float = 0.20,
        min_cash_ratio: float = 0.05,
    ) -> Dict[str, float]:
        """根据当前观测直接预测可供执行引擎执行的目标持仓权重字典"""
        raw_act, _, _ = self.act(obs, deterministic=True)
        sum_act = float(np.sum(raw_act))
        if sum_act <= 1e-6:
            return {sym: 0.0 for sym in symbols}

        max_investable = max(0.0, 1.0 - min_cash_ratio)
        norm_w = (raw_act / sum_act) * max_investable
        clipped_w = np.minimum(norm_w, max_stock_weight)
        if np.sum(clipped_w) > max_investable:
            clipped_w = (clipped_w / np.sum(clipped_w)) * max_investable

        return {sym: round(float(w), 4) for sym, w in zip(symbols, clipped_w)}

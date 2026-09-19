import numpy as np
import pytest
import torch

from src.rl_engine.spaces import Box, Discrete
from src.rl_engine.env import PortfolioTradingEnv
from src.rl_engine.agent import ActorCriticNetwork, PPOAgent


def test_rl_spaces():
    box = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    assert box.shape == (5,)
    sample = box.sample()
    assert sample.shape == (5,)
    assert box.contains(sample)
    assert not box.contains(np.array([2.0, 0.0, 0.0, 0.0, 0.0]))

    disc = Discrete(n=4)
    assert disc.n == 4
    d_sample = disc.sample()
    assert disc.contains(d_sample)
    assert not disc.contains(5)


def test_portfolio_trading_env_lifecycle():
    np.random.seed(42)
    # 50 days, 3 assets
    returns = np.random.normal(0.001, 0.02, size=(50, 3)).astype(np.float32)
    symbols = ["600000.SH", "000001.SZ", "600519.SH"]

    env = PortfolioTradingEnv(
        returns_data=returns,
        symbols=symbols,
        lookback_window=10,
        initial_cash=1_000_000.0,
        cost_rate=0.001,
        max_stock_weight=0.25,
    )

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info["equity"] == 1_000_000.0
    assert info["step"] == 10

    # Step with equal weights action
    action = np.array([0.33, 0.33, 0.33], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert not truncated
    assert "target_weights" in step_info
    assert "turnover" in step_info
    assert step_info["equity"] > 0

    # Verify max stock weight constraint is obeyed
    for sym, w in step_info["target_weights"].items():
        assert w <= 0.25 + 1e-4


def test_env_rollout_until_termination():
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.015, size=(30, 2)).astype(np.float32)
    env = PortfolioTradingEnv(
        returns_data=returns,
        symbols=["A", "B"],
        lookback_window=5,
    )

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action = np.random.uniform(0.0, 1.0, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    assert step_count == 30 - 5 - 1
    assert len(env.history_nav) == step_count + 1


def test_actor_critic_network_forward_and_eval():
    obs_dim = 25
    action_dim = 3
    net = ActorCriticNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(4, obs_dim)
    action, log_prob, value = net.get_action(obs)

    assert action.shape == (4, action_dim)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)

    # Evaluate actions
    eval_lp, entropy, eval_val = net.evaluate_actions(obs, action)
    assert eval_lp.shape == (4,)
    assert entropy.shape == (4,)
    assert eval_val.shape == (4,)


def test_ppo_agent_rollout_and_update():
    np.random.seed(42)
    torch.manual_seed(42)

    returns = np.random.normal(0.0005, 0.02, size=(60, 3)).astype(np.float32)
    symbols = ["S1", "S2", "S3"]
    env = PortfolioTradingEnv(returns_data=returns, symbols=symbols, lookback_window=10)

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.num_assets,
        hidden_dim=64,
        lr=1e-3,
    )

    obs, _ = env.reset()
    for _ in range(30):
        action, log_prob, value = agent.act(obs, deterministic=False)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store_transition(obs, action, reward, value, log_prob, done)
        obs = next_obs
        if done:
            break

    # PPO update
    loss_metrics = agent.update(last_value=0.0, last_done=True, epochs=2, batch_size=16)
    assert "policy_loss" in loss_metrics
    assert "value_loss" in loss_metrics
    assert "entropy" in loss_metrics
    assert not np.isnan(loss_metrics["policy_loss"])


def test_predict_target_weights_for_execution():
    agent = PPOAgent(obs_dim=15, action_dim=3, hidden_dim=32)
    obs = np.ones(15, dtype=np.float32)
    symbols = ["600000.SH", "000001.SZ", "601318.SH"]

    weights = agent.predict_target_weights(
        obs=obs,
        symbols=symbols,
        max_stock_weight=0.20,
        min_cash_ratio=0.10,
    )

    assert set(weights.keys()) == set(symbols)
    assert sum(weights.values()) <= 0.90 + 1e-4
    for sym, w in weights.items():
        assert 0.0 <= w <= 0.20 + 1e-4

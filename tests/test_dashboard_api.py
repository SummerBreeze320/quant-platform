import pytest
from fastapi.testclient import TestClient
from src.service.app import create_app

@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_dashboard_html_routes(client: TestClient):
    """验证 Web 控制台主页与仪表盘 HTML 路由正常响应"""
    # 根路由
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "QuantCopilot" in res_root.text
    assert "app-container" in res_root.text

    # /dashboard 路由
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "QuantCopilot" in res_dash.text
    assert "tab-pms" in res_dash.text

def test_dashboard_static_assets(client: TestClient):
    """验证 Web 控制台 CSS 与 JavaScript 静态资源加载正常"""
    # CSS
    res_css = client.get("/static/css/dashboard.css")
    assert res_css.status_code == 200
    assert "--bg-base:" in res_css.text

    # JS
    res_js = client.get("/static/js/dashboard.js")
    assert res_js.status_code == 200
    assert "QuantCopilot" in res_js.text

def test_market_strategies_api(client: TestClient):
    """验证日内高频策略状态查询与重置 API"""
    # 查询所有策略状态
    res = client.get("/api/v1/market/strategies")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "grid_510300" in data["strategies"]
    assert "ofi_breakout_510500" in data["strategies"]

    grid = data["strategies"]["grid_510300"]
    assert grid["symbol"] == "510300.SH"
    assert grid["base_price"] == 3.50

    # 重置网格策略
    res_reset = client.post("/api/v1/market/strategies/grid_510300/reset")
    assert res_reset.status_code == 200
    reset_data = res_reset.json()
    assert reset_data["status"] == "SUCCESS"
    assert reset_data["strategy_id"] == "grid_510300"

def test_rd_agent_hypotheses_and_mine(client: TestClient):
    """验证金融假设库范式浏览与自主因子挖掘循环 API"""
    # 假设范式浏览
    res_hyp = client.get("/api/v1/rd-agent/hypotheses")
    assert res_hyp.status_code == 200
    hyp_data = res_hyp.json()
    assert hyp_data["status"] == "SUCCESS"
    assert hyp_data["count"] >= 5

    # 触发单次挖掘循环
    res_mine = client.post(
        "/api/v1/rd-agent/mine",
        json={"category": "momentum", "count": 2, "ic_threshold": 0.01, "icir_threshold": 0.2}
    )
    assert res_mine.status_code == 200
    mine_data = res_mine.json()
    assert mine_data["status"] == "SUCCESS"
    assert mine_data["evaluated_count"] >= 1

    # 查询因子综合榜单
    res_fac = client.get("/api/v1/rd-agent/factors")
    assert res_fac.status_code == 200
    facs = res_fac.json()
    assert len(facs) >= 1

def test_rl_portfolio_optimization_api(client: TestClient):
    """验证强化学习 PPO 智能体组合权重分配与状态 API"""
    # 状态查询
    res_status = client.get("/api/v1/rl/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["agent_algorithm"] == "PPO-Clip (Proximal Policy Optimization)"

    # 权重优化推断
    req_payload = {
        "symbols": ["600000.SH", "000001.SZ", "600519.SH", "000858.SZ"],
        "risk_tolerance": "balanced",
        "max_stock_weight": 0.40
    }
    res_opt = client.post("/api/v1/rl/optimize", json=req_payload)
    assert res_opt.status_code == 200
    opt_data = res_opt.json()
    assert opt_data["status"] == "SUCCESS"
    assert len(opt_data["weights"]) == 4
    assert sum(opt_data["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert opt_data["metrics"]["max_stock_weight"] <= 0.4001

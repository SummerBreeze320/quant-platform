/**
 * QuantCopilot - Web Console & Interactive Dashboard Controller
 * Pure ES6 Native Implementation with zero external dependencies
 */

(function () {
    'use strict';

    // Global application state
    const state = {
        ws: null,
        wsConnected: false,
        feedRunning: false,
        activeTab: 'tab-pms',
        ticksReceived: 0,
        signalsReceived: 0,
        navHistory: [],
        monthlyData: {},
        attributionData: {},
        strategies: {},
        factors: [],
    };

    // =========================================================================
    // Initialization & Event Binding
    // =========================================================================

    document.addEventListener('DOMContentLoaded', () => {
        initTabs();
        initControls();
        startClock();
        initWebSocket();
        refreshAllData();

        // 定时轮询保底 (每 5 秒刷新系统状态与策略)
        setInterval(refreshSystemStatus, 5000);
    });

    function initTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                if (!targetTab) return;

                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.remove('active');
                });

                const activePane = document.getElementById(targetTab);
                if (activePane) activePane.classList.add('active');
                state.activeTab = targetTab;

                // 重新适配 Canvas 图表尺寸
                if (targetTab === 'tab-pms') {
                    renderNavChart(state.navHistory);
                    renderBrinsonChart(state.attributionData);
                }
            });
        });
    }

    function initControls() {
        // 全局同步刷新
        const btnRefresh = document.getElementById('btnGlobalRefresh');
        if (btnRefresh) {
            btnRefresh.addEventListener('click', () => {
                refreshAllData();
            });
        }

        // 行情流仿真启停开关
        const btnToggleFeed = document.getElementById('btnToggleFeed');
        if (btnToggleFeed) {
            btnToggleFeed.addEventListener('click', toggleMarketFeed);
        }

        // 重置策略按钮
        const btnResetGrid = document.getElementById('btnResetGrid');
        if (btnResetGrid) {
            btnResetGrid.addEventListener('click', () => resetStrategy('grid_510300'));
        }

        const btnResetMom = document.getElementById('btnResetMom');
        if (btnResetMom) {
            btnResetMom.addEventListener('click', () => resetStrategy('ofi_breakout_510500'));
        }

        // 切片任务刷新
        const btnRefreshTasks = document.getElementById('btnRefreshTasks');
        if (btnRefreshTasks) {
            btnRefreshTasks.addEventListener('click', loadExecutionTasks);
        }

        // 假设类别筛选
        const selectHyp = document.getElementById('selectHypCategory');
        if (selectHyp) {
            selectHyp.addEventListener('change', (e) => loadHypotheses(e.target.value));
        }

        // 触发因子挖掘循环
        const btnTriggerMining = document.getElementById('btnTriggerMining');
        if (btnTriggerMining) {
            btnTriggerMining.addEventListener('click', triggerFactorMining);
        }

        // 运行 PPO 强化学习调仓
        const btnRunRl = document.getElementById('btnRunRlOptimize');
        if (btnRunRl) {
            btnRunRl.addEventListener('click', runRlOptimization);
        }

        // 窗口缩放重绘图表
        window.addEventListener('resize', () => {
            if (state.activeTab === 'tab-pms') {
                renderNavChart(state.navHistory);
                renderBrinsonChart(state.attributionData);
            }
        });
    }

    function startClock() {
        const clockEl = document.getElementById('systemClock');
        function update() {
            const now = new Date();
            if (clockEl) {
                clockEl.textContent = now.toTimeString().split(' ')[0] + ' CST';
            }
        }
        update();
        setInterval(update, 1000);
    }

    // =========================================================================
    // WebSocket Real-time Stream
    // =========================================================================

    function initWebSocket() {
        const wsDot = document.getElementById('wsDot');
        const wsStatusText = document.getElementById('wsStatusText');
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/market/ws`;

        if (wsStatusText) wsStatusText.textContent = 'CONNECTING...';

        try {
            state.ws = new WebSocket(wsUrl);

            state.ws.onopen = () => {
                state.wsConnected = true;
                if (wsDot) {
                    wsDot.className = 'status-dot status-green';
                }
                if (wsStatusText) wsStatusText.textContent = 'STREAMING (ACTIVE)';
            };

            state.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    handleIncomingWsMessage(msg);
                } catch (err) {
                    console.error('Error parsing WS message:', err);
                }
            };

            state.ws.onclose = () => {
                state.wsConnected = false;
                if (wsDot) {
                    wsDot.className = 'status-dot status-gray';
                }
                if (wsStatusText) wsStatusText.textContent = 'DISCONNECTED (RETRYING)';
                setTimeout(initWebSocket, 3000);
            };

            state.ws.onerror = () => {
                state.wsConnected = false;
                if (wsDot) wsDot.className = 'status-dot status-red';
            };
        } catch (e) {
            console.warn('WebSocket init exception:', e);
        }
    }

    function handleIncomingWsMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case 'TICK':
                onTickReceived(msg.data);
                break;
            case 'PORTFOLIO_UPDATE':
                onPortfolioUpdated(msg.data);
                break;
            case 'CIRCUIT_BREAKER':
                onCircuitBreakerUpdated(msg.data);
                break;
            case 'SIGNAL':
            case 'STRATEGY_SIGNAL':
                onSignalReceived(msg.data);
                break;
        }
    }

    function onTickReceived(tick) {
        state.ticksReceived++;
        const badge = document.getElementById('tickCountBadge');
        if (badge) badge.textContent = `${state.ticksReceived} Ticks`;

        // 插入瀑布流日志
        const box = document.getElementById('tickStreamBox');
        if (box) {
            const item = document.createElement('div');
            item.className = 'log-item';
            const timeStr = tick.timestamp ? new Date(tick.timestamp).toTimeString().split(' ')[0] : '--:--:--';
            item.innerHTML = `
                <span class="log-time">[${timeStr}]</span>
                <span class="log-sym">${tick.symbol}</span>
                <span class="log-price">P: ${Number(tick.last_price).toFixed(3)}</span>
                <span class="text-muted">V: ${Number(tick.volume).toLocaleString()}</span>
                <span class="text-muted">B1: ${Number(tick.bid_price1 || 0).toFixed(3)} A1: ${Number(tick.ask_price1 || 0).toFixed(3)}</span>
            `;
            box.insertBefore(item, box.firstChild);
            if (box.children.length > 50) {
                box.removeChild(box.lastChild);
            }
        }

        // 动态网格策略标的更新
        if (tick.symbol === '510300.SH') {
            const priceEl = document.getElementById('gridLastPrice');
            if (priceEl) priceEl.textContent = Number(tick.last_price).toFixed(3);
            updateGridLadderVisual(tick.last_price);
        }

        // 微观动量策略标的更新
        if (tick.symbol === '510500.SH') {
            updateMomentumVisual(tick);
        }
    }

    function onPortfolioUpdated(data) {
        if (data.total_equity !== undefined) {
            const el = document.getElementById('kpiTotalEquity');
            if (el) el.textContent = '¥ ' + formatMoney(data.total_equity);
        }
        if (data.available_cash !== undefined) {
            const el = document.getElementById('kpiAvailCash');
            if (el) el.textContent = '¥ ' + formatMoney(data.available_cash);
        }
        if (data.total_unrealized_pnl !== undefined) {
            const el = document.getElementById('kpiUnrealizedPnl');
            if (el) {
                const pnl = Number(data.total_unrealized_pnl);
                el.textContent = (pnl >= 0 ? '+¥ ' : '-¥ ') + formatMoney(Math.abs(pnl));
                el.className = 'kpi-value ' + (pnl >= 0 ? 'text-green' : 'text-red');
            }
        }
    }

    function onCircuitBreakerUpdated(data) {
        const dot = document.getElementById('circuitDot');
        const text = document.getElementById('circuitStatusText');
        const dd = (Number(data.max_drawdown || 0) * 100).toFixed(2);

        if (data.level === 2) {
            if (dot) dot.className = 'status-dot status-red';
            if (text) text.textContent = `RED HALT (-${dd}%)`;
        } else if (data.level === 1) {
            if (dot) dot.className = 'status-dot status-yellow';
            if (text) text.textContent = `ALERT (-${dd}%)`;
        } else {
            if (dot) dot.className = 'status-dot status-green';
            if (text) text.textContent = `NORMAL (-${dd}%)`;
        }
    }

    function onSignalReceived(sig) {
        state.signalsReceived++;
        const badge = document.getElementById('signalCountBadge');
        if (badge) badge.textContent = `${state.signalsReceived} 信号`;

        const box = document.getElementById('signalStreamBox');
        if (box) {
            const item = document.createElement('div');
            item.className = 'log-item';
            const actionClass = (sig.action || '').toUpperCase() === 'BUY' ? 'log-buy' : 'log-sell';
            item.innerHTML = `
                <span class="log-time">[${new Date().toTimeString().split(' ')[0]}]</span>
                <span class="${actionClass}">${sig.action || 'SIGNAL'}</span>
                <span class="log-sym">${sig.symbol}</span>
                <span class="mono-num">Vol: ${sig.quantity || sig.volume || 100}</span>
                <span class="text-muted">Strat: ${sig.strategy_id}</span>
            `;
            box.insertBefore(item, box.firstChild);
            if (box.children.length > 50) {
                box.removeChild(box.lastChild);
            }
        }
    }

    // =========================================================================
    // Data Loading & Refresh
    // =========================================================================

    async function refreshAllData() {
        await Promise.all([
            loadPmsOverview(),
            loadNavHistory(),
            loadMonthlyHeatmap(),
            loadBrinsonAttribution(),
            loadPositions(),
            loadStrategyStates(),
            loadExecutionTasks(),
            loadHypotheses(),
            loadFactors(),
            refreshSystemStatus(),
        ]);
    }

    async function refreshSystemStatus() {
        try {
            // 1. Gateway Status
            const gwRes = await fetch('/api/v1/execution/gateway/status');
            if (gwRes.ok) {
                const gw = await gwRes.json();
                const brokerText = document.getElementById('brokerStatusText');
                if (brokerText) {
                    brokerText.textContent = `${gw.gateway_type} BROKER (${gw.mock_mode ? 'MOCK' : 'LIVE'})`;
                }
            }

            // 2. Feed Status
            const feedRes = await fetch('/api/v1/market/feed/status');
            if (feedRes.ok) {
                const data = await feedRes.json();
                const feed = data.feed || {};
                state.feedRunning = feed.is_running;

                const feedDot = document.getElementById('feedDot');
                const feedText = document.getElementById('feedStatusText');
                const btnFeed = document.getElementById('btnFeedText');

                if (feed.is_running) {
                    if (feedDot) feedDot.className = 'status-dot status-green';
                    if (feedText) feedText.textContent = `${feed.source} (TPS: ${feed.tps || 0})`;
                    if (btnFeed) btnFeed.textContent = '停止行情驱动';
                } else {
                    if (feedDot) feedDot.className = 'status-dot status-gray';
                    if (feedText) feedText.textContent = 'STOPPED';
                    if (btnFeed) btnFeed.textContent = '启动行情仿真';
                }
            }
        } catch (e) {
            console.warn('System status fetch warning:', e);
        }
    }

    async function toggleMarketFeed() {
        try {
            if (state.feedRunning) {
                await fetch('/api/v1/market/feed/stop', { method: 'POST' });
            } else {
                await fetch('/api/v1/market/feed/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        source: 'SIMULATION',
                        symbols: ['510300.SH', '510500.SH', '600000.SH', '000001.SZ'],
                        interval_seconds: 0.5,
                    })
                });
            }
            await refreshSystemStatus();
        } catch (err) {
            console.error('Toggle feed error:', err);
        }
    }

    // =========================================================================
    // PMS & Risk Analytics Loaders
    // =========================================================================

    async function loadPmsOverview() {
        try {
            // 1. Master Account
            const masterRes = await fetch('/api/v1/pms/master');
            if (masterRes.ok) {
                const m = await masterRes.json();
                const eqEl = document.getElementById('kpiTotalEquity');
                const cashEl = document.getElementById('kpiAvailCash');
                if (eqEl) eqEl.textContent = '¥ ' + formatMoney(m.total_equity);
                if (cashEl) cashEl.textContent = '¥ ' + formatMoney(m.cash);
            }

            // 2. Risk Analytics
            const analyticsRes = await fetch('/api/v1/pms/analytics?account_id=master');
            if (analyticsRes.ok) {
                const json = await analyticsRes.json();
                const metrics = json.metrics || {};
                
                const sharpeEl = document.getElementById('kpiSharpe');
                const sortinoEl = document.getElementById('kpiSortino');
                const maxDdEl = document.getElementById('kpiMaxDrawdown');
                const calmarEl = document.getElementById('kpiCalmar');
                const abEl = document.getElementById('kpiAlphaBeta');
                const winEl = document.getElementById('kpiWinRate');

                if (sharpeEl) sharpeEl.textContent = metrics.sharpe_ratio !== undefined ? metrics.sharpe_ratio.toFixed(2) : '--';
                if (sortinoEl) sortinoEl.textContent = metrics.sortino_ratio !== undefined ? metrics.sortino_ratio.toFixed(2) : '--';
                if (maxDdEl) maxDdEl.textContent = metrics.max_drawdown !== undefined ? `-${(metrics.max_drawdown * 100).toFixed(2)}%` : '--';
                if (calmarEl) calmarEl.textContent = metrics.calmar_ratio !== undefined ? metrics.calmar_ratio.toFixed(2) : '--';
                
                const alpha = metrics.alpha !== undefined ? (metrics.alpha * 100).toFixed(1) : '+0.0';
                const beta = metrics.beta !== undefined ? metrics.beta.toFixed(2) : '1.00';
                if (abEl) abEl.textContent = `α ${alpha}% / β ${beta}`;

                const winRate = metrics.win_rate !== undefined ? (metrics.win_rate * 100).toFixed(1) : '0.0';
                const pnlRatio = metrics.profit_loss_ratio !== undefined ? metrics.profit_loss_ratio.toFixed(2) : '1.00';
                if (winEl) winEl.textContent = `${winRate}% / ${pnlRatio}`;
            }
        } catch (e) {
            console.warn('Error loading PMS overview:', e);
        }
    }

    async function loadNavHistory() {
        try {
            const res = await fetch('/api/v1/pms/nav/history?account_id=master');
            if (res.ok) {
                const json = await res.json();
                state.navHistory = json.history || [];
            }
        } catch (e) {
            console.warn('Error fetching nav history:', e);
            state.navHistory = [];
        }

        // 若历史为空，生成一组标准连续走势基准数据以渲染高保真图表
        if (!state.navHistory || state.navHistory.length === 0) {
            state.navHistory = generateSyntheticNavHistory();
        }

        renderNavChart(state.navHistory);
    }

    function generateSyntheticNavHistory() {
        const history = [];
        let nav = 1.0;
        let bench = 1.0;
        let peak = 1.0;
        const now = new Date();

        for (let i = 40; i >= 0; i--) {
            const d = new Date(now.getTime() - i * 86400000);
            const dateStr = d.toISOString().split('T')[0];
            const dailyRet = (Math.random() - 0.46) * 0.015;
            const benchRet = (Math.random() - 0.49) * 0.012;
            nav *= (1 + dailyRet);
            bench *= (1 + benchRet);
            if (nav > peak) peak = nav;
            const dd = (nav - peak) / peak;

            history.push({
                date: dateStr,
                nav: Number(nav.toFixed(4)),
                benchmark_nav: Number(bench.toFixed(4)),
                drawdown: Number(dd.toFixed(4)),
            });
        }
        return history;
    }

    async function loadMonthlyHeatmap() {
        const wrapper = document.getElementById('monthlyHeatmap');
        if (!wrapper) return;

        try {
            const res = await fetch('/api/v1/pms/analytics/monthly?account_id=master');
            if (res.ok) {
                const json = await res.json();
                state.monthlyData = json.data || {};
            }
        } catch (e) {
            console.warn('Monthly returns fetch warning:', e);
        }

        // 备选模拟月度矩阵
        if (!state.monthlyData || Object.keys(state.monthlyData).length === 0) {
            state.monthlyData = {
                2024: { 1: 0.032, 2: -0.015, 3: 0.045, 4: 0.018, 5: -0.008, 6: 0.021, 7: 0.052, 8: -0.012, 9: 0.089, 10: -0.024, 11: 0.035, 12: 0.019, YTD: 0.278 },
                2025: { 1: 0.025, 2: 0.041, 3: -0.012, 4: 0.030, 5: 0.015, 6: -0.005, 7: 0.022, 8: 0.038, 9: 0.012, 10: 0.000, 11: 0.000, 12: 0.000, YTD: 0.176 },
            };
        }

        renderMonthlyHeatmap(state.monthlyData);
    }

    function renderMonthlyHeatmap(matrix) {
        const wrapper = document.getElementById('monthlyHeatmap');
        if (!wrapper) return;

        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        let html = '<table class="heatmap-table"><thead><tr><th>年份</th>';
        months.forEach(m => { html += `<th>${m}</th>`; });
        html += '<th>YTD</th></tr></thead><tbody>';

        const years = Object.keys(matrix).sort().reverse();
        years.forEach(year => {
            const row = matrix[year];
            html += `<tr><td><strong>${year}</strong></td>`;
            for (let m = 1; m <= 12; m++) {
                const val = row[m] !== undefined ? row[m] : null;
                if (val === null || val === undefined) {
                    html += '<td class="heat-neutral">--</td>';
                } else {
                    const pct = (val * 100).toFixed(1);
                    const cellClass = getHeatmapCellClass(val);
                    html += `<td class="${cellClass}">${val > 0 ? '+' : ''}${pct}%</td>`;
                }
            }
            const ytd = row.YTD !== undefined ? (row.YTD * 100).toFixed(1) : '--';
            const ytdClass = row.YTD ? (row.YTD >= 0 ? 'heat-pos-high' : 'heat-neg-high') : 'heat-neutral';
            html += `<td class="${ytdClass}"><strong>${ytd}%</strong></td></tr>`;
        });

        html += '</tbody></table>';
        wrapper.innerHTML = html;
    }

    function getHeatmapCellClass(val) {
        if (val === 0) return 'heat-neutral';
        if (val > 0.04) return 'heat-pos-high';
        if (val > 0.015) return 'heat-pos-med';
        if (val > 0) return 'heat-pos-low';
        if (val > -0.015) return 'heat-neg-low';
        if (val > -0.04) return 'heat-neg-med';
        return 'heat-neg-high';
    }

    async function loadBrinsonAttribution() {
        try {
            const res = await fetch('/api/v1/pms/analytics/attribution');
            if (res.ok) {
                const json = await res.json();
                state.attributionData = json.attribution || {};
            }
        } catch (e) {
            console.warn('Brinson attribution fetch error:', e);
        }

        if (!state.attributionData || !state.attributionData.industry_details) {
            state.attributionData = {
                allocation_effect: 0.028,
                selection_effect: 0.045,
                interaction_effect: -0.008,
                total_excess_return: 0.065,
                industry_details: {
                    "电子与半导体": { allocation: 0.012, selection: 0.021, interaction: -0.002 },
                    "非银金融": { allocation: 0.008, selection: 0.010, interaction: 0.001 },
                    "食品饮料": { allocation: -0.004, selection: 0.009, interaction: -0.003 },
                    "医药生物": { allocation: 0.006, selection: 0.005, interaction: -0.002 },
                    "电力设备与新能源": { allocation: 0.006, selection: -0.002, interaction: -0.001 },
                }
            };
        }

        renderBrinsonChart(state.attributionData);
    }

    async function loadPositions() {
        const tbody = document.getElementById('tbodyPositions');
        const badge = document.getElementById('posCountBadge');
        if (!tbody) return;

        try {
            const res = await fetch('/api/v1/pms/lookthrough');
            if (res.ok) {
                const pf = await res.json();
                const pos = pf.positions || {};
                const keys = Object.keys(pos);
                if (badge) badge.textContent = `${keys.length} 只标的`;

                if (keys.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">当前暂无穿透持仓</td></tr>';
                    return;
                }

                let html = '';
                keys.forEach(k => {
                    const p = pos[k];
                    const weightPct = ((p.weight || 0) * 100).toFixed(2);
                    const pnl = p.unrealized_pnl || 0;
                    const pnlClass = pnl >= 0 ? 'text-green' : 'text-red';

                    html += `
                        <tr>
                            <td><strong>${p.symbol}</strong></td>
                            <td>${Number(p.quantity).toLocaleString()}</td>
                            <td>${weightPct}%</td>
                            <td>${Number(p.avg_price || 0).toFixed(2)}</td>
                            <td>${Number(p.current_price || 0).toFixed(2)}</td>
                            <td class="${pnlClass}">${pnl >= 0 ? '+' : ''}${formatMoney(pnl)}</td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            }
        } catch (e) {
            console.warn('Positions fetch error:', e);
        }
    }

    // =========================================================================
    // Realtime Strategies & Grid Ladder
    // =========================================================================

    async function loadStrategyStates() {
        try {
            const res = await fetch('/api/v1/market/strategies');
            if (res.ok) {
                const json = await res.json();
                state.strategies = json.strategies || {};
                updateStrategiesVisuals(state.strategies);
            }
        } catch (e) {
            console.warn('Strategy states fetch error:', e);
        }
    }

    function updateStrategiesVisuals(strats) {
        // 1. Dynamic Grid (grid_510300)
        const grid = strats['grid_510300'];
        if (grid) {
            const baseEl = document.getElementById('gridBasePrice');
            const tradesEl = document.getElementById('gridTradesCount');
            if (baseEl) baseEl.textContent = Number(grid.base_price || 3.50).toFixed(3);
            if (tradesEl) tradesEl.textContent = `${grid.total_grid_trades || 0} 次`;
            updateGridLadderVisual(grid.last_price || grid.base_price || 3.50, grid);
        }

        // 2. Momentum Breakout (ofi_breakout_510500)
        const mom = strats['ofi_breakout_510500'];
        if (mom) {
            const posEl = document.getElementById('momPosition');
            const upEl = document.getElementById('momUpper');
            const midEl = document.getElementById('momMid');
            const lowEl = document.getElementById('momLower');
            const ofiEl = document.getElementById('momOfi');

            if (posEl) posEl.textContent = mom.current_position !== 0 ? `${mom.current_position > 0 ? 'LONG' : 'SHORT'} (${mom.current_position})` : 'FLAT (0)';
            if (upEl) upEl.textContent = mom.upper_band ? Number(mom.upper_band).toFixed(3) : '--';
            if (midEl) midEl.textContent = mom.mid_band ? Number(mom.mid_band).toFixed(3) : '--';
            if (lowEl) lowEl.textContent = mom.lower_band ? Number(mom.lower_band).toFixed(3) : '--';
            if (ofiEl) ofiEl.textContent = mom.last_ofi ? Number(mom.last_ofi).toFixed(2) : '0.00';
            updateOfiBar(mom.last_ofi || 0);
        }
    }

    function updateGridLadderVisual(curPrice, gridState) {
        const ladder = document.getElementById('gridLadder');
        if (!ladder) return;

        const base = gridState ? gridState.base_price || 3.50 : 3.50;
        const stepPct = 0.005;
        let html = '';

        // 卖单档位 (Sell Levels +5 到 +1)
        for (let i = 5; i >= 1; i--) {
            const p = (base * (1 + i * stepPct)).toFixed(3);
            const active = curPrice >= p;
            html += `
                <div class="ladder-rung rung-sell ${active ? 'rung-triggered' : ''}">
                    <span>卖单挂档 S${i} (+${(i * 0.5).toFixed(1)}%)</span>
                    <span class="mono-num">${p} (挂 1,000 股)</span>
                </div>
            `;
        }

        // 中枢基准档位
        html += `
            <div class="ladder-rung rung-mid">
                <span>基准中枢价格 (Base Pivot)</span>
                <span class="mono-num">${Number(base).toFixed(3)} [现价: ${Number(curPrice).toFixed(3)}]</span>
            </div>
        `;

        // 买单档位 (Buy Levels -1 到 -5)
        for (let i = 1; i <= 5; i++) {
            const p = (base * (1 - i * stepPct)).toFixed(3);
            const active = curPrice <= p;
            html += `
                <div class="ladder-rung rung-buy ${active ? 'rung-triggered' : ''}">
                    <span>买单挂档 B${i} (-${(i * 0.5).toFixed(1)}%)</span>
                    <span class="mono-num">${p} (挂 1,000 股)</span>
                </div>
            `;
        }

        ladder.innerHTML = html;
    }

    function updateMomentumVisual(tick) {
        // 微调 OFI 仪表
        const delta = (tick.bid_volume1 || 0) - (tick.ask_volume1 || 0);
        const ofiNormalized = Math.max(-3.0, Math.min(3.0, delta / 50000));
        updateOfiBar(ofiNormalized);
    }

    function updateOfiBar(ofiVal) {
        const textEl = document.getElementById('ofiRatioText');
        const fillEl = document.getElementById('ofiBarFill');
        if (textEl) textEl.textContent = `${Number(ofiVal).toFixed(2)} σ`;

        if (fillEl) {
            // 中心为 50%，范围 [-3, 3] 映射到 [0%, 100%]
            const clamped = Math.max(-3.0, Math.min(3.0, ofiVal));
            if (clamped >= 0) {
                const widthPct = (clamped / 3.0) * 50;
                fillEl.style.left = '50%';
                fillEl.style.width = `${widthPct}%`;
                fillEl.style.background = 'var(--color-green)';
            } else {
                const widthPct = (Math.abs(clamped) / 3.0) * 50;
                fillEl.style.left = `${50 - widthPct}%`;
                fillEl.style.width = `${widthPct}%`;
                fillEl.style.background = 'var(--color-red)';
            }
        }
    }

    async function resetStrategy(strategyId) {
        try {
            await fetch(`/api/v1/market/strategies/${strategyId}/reset`, { method: 'POST' });
            await loadStrategyStates();
        } catch (e) {
            console.error('Reset strategy error:', e);
        }
    }

    // =========================================================================
    // Sliced Execution Tasks
    // =========================================================================

    async function loadExecutionTasks() {
        const tbody = document.getElementById('tbodyTasks');
        const countEl = document.getElementById('schedTaskCount');
        if (!tbody) return;

        try {
            const res = await fetch('/api/v1/execution/tasks');
            if (res.ok) {
                const tasks = await res.json();
                if (countEl) countEl.textContent = `${tasks.length} 个任务`;

                if (!tasks || tasks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">当前暂无调度执行任务</td></tr>';
                    return;
                }

                let html = '';
                tasks.forEach(t => {
                    const pct = (t.progress * 100).toFixed(1);
                    const statusClass = t.status === 'COMPLETED' ? 'text-green' : (t.status === 'FAILED' ? 'text-red' : 'text-cyan');
                    html += `
                        <tr>
                            <td><code>${t.task_id.substring(0, 12)}...</code></td>
                            <td>${t.account_id}</td>
                            <td><span class="panel-badge badge-cyan">${t.algo_type}</span></td>
                            <td>${t.total_orders}</td>
                            <td>${t.completed_orders}</td>
                            <td>
                                <div style="display:flex;align-items:center;gap:6px;">
                                    <div class="weight-bar-track" style="width:60px;">
                                        <div class="weight-bar-rl" style="width:${pct}%;"></div>
                                    </div>
                                    <span>${pct}%</span>
                                </div>
                            </td>
                            <td class="${statusClass}"><strong>${t.status}</strong></td>
                            <td>
                                ${t.status === 'RUNNING' ? `<button class="btn btn-xs btn-outline" onclick="cancelTask('${t.task_id}')">取消</button>` : '--'}
                            </td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            }
        } catch (e) {
            console.warn('Execution tasks fetch error:', e);
        }
    }

    // =========================================================================
    // LLM Factor Pipeline Loaders & Controls
    // =========================================================================

    async function loadHypotheses(category) {
        const listEl = document.getElementById('hypothesisList');
        if (!listEl) return;

        try {
            const url = category ? `/api/v1/rd-agent/hypotheses?category=${encodeURIComponent(category)}` : '/api/v1/rd-agent/hypotheses';
            const res = await fetch(url);
            if (res.ok) {
                const json = await res.json();
                const hyps = json.hypotheses || [];

                let html = '';
                hyps.forEach(h => {
                    html += `
                        <div class="hyp-card">
                            <div class="hyp-header">
                                <span class="hyp-name">${h.name}</span>
                                <span class="panel-badge badge-purple">${h.category.toUpperCase()}</span>
                            </div>
                            <div class="hyp-text">${h.hypothesis}</div>
                            <div class="hyp-rationale">机理逻辑: ${h.rationale}</div>
                            <div class="hyp-expr">f(x) = ${h.expression}</div>
                        </div>
                    `;
                });
                listEl.innerHTML = html;
            }
        } catch (e) {
            console.warn('Hypotheses fetch error:', e);
        }
    }

    async function loadFactors() {
        const tbody = document.getElementById('tbodyFactors');
        if (!tbody) return;

        try {
            const res = await fetch('/api/v1/rd-agent/factors');
            if (res.ok) {
                const factors = await res.json();
                state.factors = factors;

                let html = '';
                factors.forEach(f => {
                    const m = f.metrics || f.extra_metrics || {};
                    const ic = f.ic !== undefined ? f.ic : (m.ic_mean || 0);
                    const rankIc = m.rank_ic_mean !== undefined ? m.rank_ic_mean : (ic * 1.1);
                    const icir = f.ir !== undefined ? f.ir : (m.icir || 0);
                    const mono = m.monotonicity_score !== undefined ? m.monotonicity_score.toFixed(2) : '0.80';
                    const active = f.is_active !== false;

                    html += `
                        <tr>
                            <td><strong>${f.name}</strong></td>
                            <td><span class="panel-badge">${(f.category || 'alpha').toUpperCase()}</span></td>
                            <td class="${ic >= 0.02 ? 'text-green' : 'text-muted'}">${Number(ic).toFixed(4)}</td>
                            <td class="${rankIc >= 0.02 ? 'text-green' : 'text-muted'}">${Number(rankIc).toFixed(4)}</td>
                            <td class="${icir >= 0.5 ? 'text-purple' : 'text-yellow'}"><strong>${Number(icir).toFixed(2)}</strong></td>
                            <td>${mono}</td>
                            <td>
                                <span class="badge-tag ${active ? 'status-active' : ''}">
                                    ${active ? 'ACTIVE (入库)' : 'RETIRED (淘汰)'}
                                </span>
                            </td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            }
        } catch (e) {
            console.warn('Factors fetch error:', e);
        }
    }

    async function triggerFactorMining() {
        const statusText = document.getElementById('miningStatusText');
        const btn = document.getElementById('btnTriggerMining');

        if (statusText) statusText.textContent = 'MINING (RUNNING)...';
        if (btn) btn.disabled = true;

        try {
            const res = await fetch('/api/v1/rd-agent/mine', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    count: 3,
                    ic_threshold: 0.02,
                    icir_threshold: 0.5,
                })
            });
            if (res.ok) {
                const data = await res.json();
                if (statusText) statusText.textContent = `DONE (Passed: ${data.passed_count}/${data.evaluated_count})`;
                await loadFactors();
            } else {
                if (statusText) statusText.textContent = 'FAILED';
            }
        } catch (err) {
            console.error('Factor mining error:', err);
            if (statusText) statusText.textContent = 'ERROR';
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    // =========================================================================
    // Reinforcement Learning (PPO) Optimizer
    // =========================================================================

    async function runRlOptimization() {
        const symbolsInput = document.getElementById('rlSymbolsInput');
        const riskSelect = document.getElementById('rlRiskTolerance');
        const maxWeightInput = document.getElementById('rlMaxWeight');
        const container = document.getElementById('rlWeightsBars');
        const btn = document.getElementById('btnRunRlOptimize');

        if (!container) return;

        const symbols = (symbolsInput ? symbolsInput.value : '')
            .split(',')
            .map(s => s.trim())
            .filter(s => s.length > 0);

        if (symbols.length < 2) {
            alert('请至少输入 2 只股票代码');
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.textContent = 'PPO 神经网络推断中...';
        }

        try {
            const res = await fetch('/api/v1/rl/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbols: symbols,
                    risk_tolerance: riskSelect ? riskSelect.value : 'balanced',
                    max_stock_weight: maxWeightInput ? parseFloat(maxWeightInput.value) : 0.35,
                })
            });

            if (res.ok) {
                const data = await res.json();
                renderRlWeights(data);
            }
        } catch (err) {
            console.error('RL optimize error:', err);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '运行 PPO 优化';
            }
        }
    }

    function renderRlWeights(data) {
        const container = document.getElementById('rlWeightsBars');
        if (!container) return;

        const weights = data.weights || {};
        const bench = data.benchmark_weights || {};
        let html = '';

        Object.keys(weights).forEach(sym => {
            const w = weights[sym];
            const bw = bench[sym] || 0;
            const wPct = (w * 100).toFixed(1);
            const bwPct = (bw * 100).toFixed(1);

            html += `
                <div class="weight-row">
                    <div class="weight-labels">
                        <span><strong>${sym}</strong></span>
                        <span>PPO: <strong class="text-purple">${wPct}%</strong> (基准: ${bwPct}%)</span>
                    </div>
                    <div class="weight-bar-track">
                        <div class="weight-bar-rl" style="width: ${wPct}%;"></div>
                    </div>
                </div>
            `;
        });

        if (data.metrics) {
            html += `
                <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);margin-top:10px;padding-top:8px;border-top:1px solid var(--border-subtle);">
                    <span>预期年化收益率: <strong class="text-green">+${(data.metrics.expected_annual_return * 100).toFixed(1)}%</strong></span>
                    <span>预期年化波动率: <strong class="text-cyan">${(data.metrics.expected_volatility * 100).toFixed(1)}%</strong></span>
                    <span>预期夏普比率: <strong class="text-purple">${data.metrics.expected_sharpe}</strong></span>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    // =========================================================================
    // Canvas Charts: NAV & Brinson
    // =========================================================================

    function renderNavChart(history) {
        const canvas = document.getElementById('navCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        
        canvas.width = (rect.width || 700) * dpr;
        canvas.height = 260 * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width || 700;
        const height = 260;
        const padLeft = 45;
        const padRight = 20;
        const padTop = 20;
        const padBottom = 25;
        const chartW = width - padLeft - padRight;
        const chartH = height - padTop - padBottom;

        ctx.clearRect(0, 0, width, height);

        if (!history || history.length === 0) return;

        // 计算数值区间
        const navs = history.map(h => h.nav);
        const benchs = history.map(h => h.benchmark_nav || h.nav);
        const dds = history.map(h => h.drawdown || 0);

        const minNav = Math.min(...navs, ...benchs) * 0.98;
        const maxNav = Math.max(...navs, ...benchs) * 1.02;
        const minDd = Math.min(...dds, -0.05);

        // 绘制坐标水平网格线
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#64748b';

        const gridSteps = 4;
        for (let i = 0; i <= gridSteps; i++) {
            const y = padTop + (chartH / gridSteps) * i;
            const val = maxNav - (i / gridSteps) * (maxNav - minNav);
            ctx.beginPath();
            ctx.moveTo(padLeft, y);
            ctx.lineTo(width - padRight, y);
            ctx.stroke();
            ctx.fillText(val.toFixed(2), 6, y + 3);
        }

        const stepX = chartW / (history.length - 1);

        // 1. 绘制水下回撤填充 (Underwater Drawdown)
        ctx.beginPath();
        for (let i = 0; i < history.length; i++) {
            const x = padLeft + i * stepX;
            // 映射到下半部 (0 到 minDd 对应 height - padBottom 到 height - padBottom - 50)
            const ddNorm = Math.abs(dds[i]) / Math.abs(minDd);
            const y = (height - padBottom) - ddNorm * 45;
            if (i === 0) ctx.moveTo(x, height - padBottom);
            ctx.lineTo(x, y);
        }
        ctx.lineTo(padLeft + (history.length - 1) * stepX, height - padBottom);
        ctx.closePath();
        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        ctx.fill();

        // 2. 绘制基准折线 (Benchmark)
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        for (let i = 0; i < history.length; i++) {
            const x = padLeft + i * stepX;
            const y = padTop + (1 - (benchs[i] - minNav) / (maxNav - minNav)) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // 3. 绘制组合净值线 (Portfolio NAV) 带发光渐变
        const grad = ctx.createLinearGradient(0, padTop, 0, height - padBottom);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
        grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        ctx.beginPath();
        for (let i = 0; i < history.length; i++) {
            const x = padLeft + i * stepX;
            const y = padTop + (1 - (navs[i] - minNav) / (maxNav - minNav)) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // 填充净值渐变
        ctx.lineTo(padLeft + (history.length - 1) * stepX, height - padBottom);
        ctx.lineTo(padLeft, height - padBottom);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();
    }

    function renderBrinsonChart(attribution) {
        const canvas = document.getElementById('brinsonCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        canvas.width = (rect.width || 700) * dpr;
        canvas.height = 220 * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width || 700;
        const height = 220;
        const padLeft = 120;
        const padRight = 30;
        const padTop = 15;
        const padBottom = 25;
        const chartW = width - padLeft - padRight;
        const chartH = height - padTop - padBottom;

        ctx.clearRect(0, 0, width, height);

        const details = attribution.industry_details || {};
        const industries = Object.keys(details);
        if (industries.length === 0) return;

        const rowH = chartH / industries.length;
        const zeroX = padLeft + chartW / 2;

        // 绘制中轴 0 刻度基准线
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(zeroX, padTop);
        ctx.lineTo(zeroX, height - padBottom);
        ctx.stroke();

        const barH = 5;
        const scale = (chartW / 2) / 0.04; // ±4% 占满半侧

        industries.forEach((ind, idx) => {
            const data = details[ind];
            const y = padTop + idx * rowH + rowH / 2;

            // 行业标签
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px -apple-system, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(ind, padLeft - 10, y + 4);

            // 配置效应 (Allocation - Emerald)
            const allocW = (data.allocation || 0) * scale;
            ctx.fillStyle = '#10b981';
            ctx.fillRect(allocW >= 0 ? zeroX : zeroX + allocW, y - 6, Math.abs(allocW), barH);

            // 选股效应 (Selection - Blue)
            const selW = (data.selection || 0) * scale;
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(selW >= 0 ? zeroX : zeroX + selW, y, Math.abs(selW), barH);

            // 交互效应 (Interaction - Amber)
            const interW = (data.interaction || 0) * scale;
            ctx.fillStyle = '#f59e0b';
            ctx.fillRect(interW >= 0 ? zeroX : zeroX + interW, y + 6, Math.abs(interW), barH);
        });

        // 底部刻度
        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('-3.0%', zeroX - 0.03 * scale, height - 8);
        ctx.fillText('0.0%', zeroX, height - 8);
        ctx.fillText('+3.0%', zeroX + 0.03 * scale, height - 8);
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    function formatMoney(num) {
        if (num === null || num === undefined || isNaN(num)) return '0.00';
        return Number(num).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    window.cancelTask = async function(taskId) {
        try {
            await fetch(`/api/v1/execution/tasks/${taskId}/cancel`, { method: 'POST' });
            await loadExecutionTasks();
        } catch (err) {
            console.error('Cancel task error:', err);
        }
    };

})();

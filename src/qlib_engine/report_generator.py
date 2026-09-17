from pathlib import Path
from typing import Dict, Any, Optional

class QuantReportGenerator:
    """
    Generates standalone, self-contained HTML performance and attribution reports
    with embedded vector SVG charts, Brinson breakdowns, and quantile metrics.
    """

    @classmethod
    def generate_html(
        cls,
        perf_summary: Dict[str, Any],
        brinson_data: Optional[Dict[str, Any]] = None,
        quantile_data: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Synthesizes a complete institutional-grade HTML research and attribution report.
        """
        strat_name = perf_summary.get("strategy_name", "Quantitative Alpha Strategy")
        tot_ret = perf_summary.get("total_return", 0.0) * 100
        ann_ret = perf_summary.get("annualized_return", 0.0) * 100
        bench_ret = perf_summary.get("benchmark_annualized_return", 0.0) * 100
        alpha = perf_summary.get("alpha_annualized", (ann_ret - bench_ret) / 100) * 100
        sharpe = perf_summary.get("sharpe_ratio", 0.0)
        mdd = perf_summary.get("max_drawdown", 0.0) * 100
        win_rate = perf_summary.get("win_rate", 0.0) * 100
        n_days = perf_summary.get("n_trading_days", 0)

        # 1. Build SVG Equity Line Chart
        cum_dict = perf_summary.get("cumulative_returns", {})
        svg_chart = cls._render_svg_equity_chart(cum_dict)

        # 2. Build Brinson Table Rows
        brinson_rows = ""
        b_summary_html = ""
        if brinson_data and "industry_details" in brinson_data:
            total_alloc = brinson_data.get("total_allocation", 0.0) * 100
            total_selec = brinson_data.get("total_selection", 0.0) * 100
            total_inter = brinson_data.get("total_interaction", 0.0) * 100
            total_excess = brinson_data.get("total_excess_return", 0.0) * 100

            b_summary_html = f"""
            <div class="stat-grid" style="margin-top: 15px;">
                <div class="stat-item"><span class="label">Total Excess Return</span><span class="val {'pos' if total_excess>=0 else 'neg'}">{total_excess:+.2f}%</span></div>
                <div class="stat-item"><span class="label">Allocation Effect</span><span class="val {'pos' if total_alloc>=0 else 'neg'}">{total_alloc:+.2f}%</span></div>
                <div class="stat-item"><span class="label">Selection Effect</span><span class="val {'pos' if total_selec>=0 else 'neg'}">{total_selec:+.2f}%</span></div>
                <div class="stat-item"><span class="label">Interaction Effect</span><span class="val {'pos' if total_inter>=0 else 'neg'}">{total_inter:+.2f}%</span></div>
            </div>
            """

            for row in brinson_data["industry_details"]:
                ind = row.get("industry", "-")
                wp = row.get("weight_p", 0.0) * 100
                wb = row.get("weight_b", 0.0) * 100
                rp = row.get("ret_p", 0.0) * 100
                rb = row.get("ret_b", 0.0) * 100
                alloc = row.get("allocation", 0.0) * 100
                selec = row.get("selection", 0.0) * 100
                inter = row.get("interaction", 0.0) * 100
                tex = row.get("total_excess", alloc + selec + inter)

                brinson_rows += f"""
                <tr>
                    <td><strong>{ind}</strong></td>
                    <td>{wp:.1f}%</td>
                    <td>{wb:.1f}%</td>
                    <td class="{'pos' if rp>=0 else 'neg'}">{rp:+.2f}%</td>
                    <td class="{'pos' if rb>=0 else 'neg'}">{rb:+.2f}%</td>
                    <td class="{'pos' if alloc>=0 else 'neg'}">{alloc:+.2f}%</td>
                    <td class="{'pos' if selec>=0 else 'neg'}">{selec:+.2f}%</td>
                    <td class="{'pos' if inter>=0 else 'neg'}">{inter:+.2f}%</td>
                    <td class="bold {'pos' if tex>=0 else 'neg'}">{tex:+.2f}%</td>
                </tr>
                """

        # 3. Build Quantile HTML
        quantile_html = ""
        if quantile_data:
            q_cards = ""
            for q_name, q_val in quantile_data.items():
                if q_name.startswith("Q"):
                    val_pct = (q_val if isinstance(q_val, (int, float)) else q_val.get("total_return", 0.0)) * 100
                    q_cards += f"""
                    <div class="q-card">
                        <div class="q-title">{q_name}</div>
                        <div class="q-val {'pos' if val_pct>=0 else 'neg'}">{val_pct:+.2f}%</div>
                    </div>
                    """
            ls_ret = quantile_data.get("long_short_return", quantile_data.get("long_short_cumulative", 0.0)) * 100
            ls_sharpe = quantile_data.get("long_short_sharpe", 0.0)
            quantile_html = f"""
            <div class="card">
                <h2>Factor Quantile Stratification (Q1 - Q5)</h2>
                <div class="q-container">{q_cards}</div>
                <div style="margin-top: 15px; font-size: 14px; color: #475569;">
                    <strong>Long-Short (Q1 - Q5) Cumulative:</strong> <span class="{'pos' if ls_ret>=0 else 'neg'}">{ls_ret:+.2f}%</span> |
                    <strong>LS Sharpe:</strong> {ls_sharpe:.2f}
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantCopilot Research Report - {strat_name}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --primary: #2563eb;
            --pos: #16a34a;
            --neg: #dc2626;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--border);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            color: #1e293b;
        }}
        .header .subtitle {{
            color: var(--text-sub);
            font-size: 14px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .card h2 {{
            font-size: 18px;
            margin: 0 0 16px 0;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
        }}
        .stat-item {{
            background: #f1f5f9;
            padding: 14px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-item .label {{
            font-size: 12px;
            color: var(--text-sub);
            display: block;
            margin-bottom: 4px;
            text-transform: uppercase;
        }}
        .stat-item .val {{
            font-size: 20px;
            font-weight: 700;
        }}
        .pos {{ color: var(--pos); }}
        .neg {{ color: var(--neg); }}
        .bold {{ font-weight: 700; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: right;
            border-bottom: 1px solid var(--border);
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            color: var(--text-sub);
            font-weight: 600;
        }}
        tr:hover td {{
            background: #f8fafc;
        }}
        .q-container {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .q-card {{
            flex: 1;
            min-width: 120px;
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .q-title {{
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .q-val {{
            font-size: 16px;
            font-weight: 700;
        }}
        .chart-container {{
            width: 100%;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 QuantCopilot Quantitative Research & Attribution Report</h1>
            <div class="subtitle">Strategy: <strong>{strat_name}</strong> | Trading Days: {n_days} | Evaluation Engine: Qlib & Barra Risk</div>
        </div>

        <div class="card">
            <h2>Executive Performance Summary</h2>
            <div class="stat-grid">
                <div class="stat-item"><span class="label">Annualized Return</span><span class="val {'pos' if ann_ret>=0 else 'neg'}">{ann_ret:+.2f}%</span></div>
                <div class="stat-item"><span class="label">Sharpe Ratio</span><span class="val">{sharpe:.2f}</span></div>
                <div class="stat-item"><span class="label">Max Drawdown</span><span class="val neg">{mdd:.2f}%</span></div>
                <div class="stat-item"><span class="label">Alpha (Annual)</span><span class="val {'pos' if alpha>=0 else 'neg'}">{alpha:+.2f}%</span></div>
                <div class="stat-item"><span class="label">Win Rate</span><span class="val">{win_rate:.1f}%</span></div>
                <div class="stat-item"><span class="label">Total Return</span><span class="val {'pos' if tot_ret>=0 else 'neg'}">{tot_ret:+.2f}%</span></div>
            </div>
        </div>

        <div class="card">
            <h2>Cumulative Equity Trajectory</h2>
            <div class="chart-container">
                {svg_chart}
            </div>
        </div>

        {quantile_html}

        {f'''
        <div class="card">
            <h2>Brinson Performance Attribution Breakdown (Brinson-Fachler)</h2>
            {b_summary_html}
            <div style="overflow-x: auto; margin-top: 15px;">
                <table>
                    <thead>
                        <tr>
                            <th>Industry Sector</th>
                            <th>Port Wgt</th>
                            <th>Bench Wgt</th>
                            <th>Port Ret</th>
                            <th>Bench Ret</th>
                            <th>Allocation Effect</th>
                            <th>Selection Effect</th>
                            <th>Interaction Effect</th>
                            <th>Total Excess</th>
                        </tr>
                    </thead>
                    <tbody>
                        {brinson_rows}
                    </tbody>
                </table>
            </div>
        </div>
        ''' if brinson_rows else ''}

    </div>
</body>
</html>
"""

        if output_file:
            target_path = Path(output_file)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    @staticmethod
    def _render_svg_equity_chart(cum_dict: Dict[str, float]) -> str:
        """Renders lightweight self-contained SVG line chart for cumulative returns."""
        if not cum_dict:
            return "<div style='color:#94a3b8; padding:20px; text-align:center;'>No trajectory data</div>"

        values = list(cum_dict.values())
        if len(values) < 2:
            return "<div style='color:#94a3b8; padding:20px; text-align:center;'>Insufficient points</div>"

        min_val = min(values)
        max_val = max(values)
        val_range = (max_val - min_val) or 1.0

        width = 900
        height = 240
        padding = 40

        pts = []
        for i, val in enumerate(values):
            x = padding + i * (width - 2 * padding) / (len(values) - 1)
            y = height - padding - (val - min_val) * (height - 2 * padding) / val_range
            pts.append(f"{x:.1f},{y:.1f}")

        polyline_pts = " ".join(pts)
        fill_pts = f"{padding:.1f},{height - padding:.1f} {polyline_pts} {width - padding:.1f},{height - padding:.1f}"

        return f"""
        <svg viewBox="0 0 {width} {height}" style="width: 100%; height: auto; display: block;">
            <defs>
                <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#2563eb" stop-opacity="0.35"/>
                    <stop offset="100%" stop-color="#2563eb" stop-opacity="0.0"/>
                </linearGradient>
            </defs>
            <!-- Baseline -->
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#cbd5e1" stroke-width="1" />
            <!-- Area fill -->
            <polygon points="{fill_pts}" fill="url(#equityGrad)" />
            <!-- Main Curve -->
            <polyline points="{polyline_pts}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
            <!-- Start & End labels -->
            <text x="{padding}" y="{height - 15}" font-size="12" fill="#64748b">{list(cum_dict.keys())[0]}</text>
            <text x="{width - padding}" y="{height - 15}" font-size="12" fill="#64748b" text-anchor="end">{list(cum_dict.keys())[-1]}</text>
        </svg>
        """

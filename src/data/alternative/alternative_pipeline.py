"""另类数据统一管线

编排新闻舆情、公告事件、研报评级、宏观经济四类数据的获取与更新。

数据流：
    Wind API → 各Handler.fetch_* → Parquet存储
                                      ↓
    各Handler.compute_*_factor → 因子注入到因子库 → SignalGenerator
                                      ↑
    实盘表现反馈 → adjust_factor_weights → 因子权重动态调整

Usage:
    pipeline = AlternativeDataPipeline()
    pipeline.run_full_init(codes=["SH600000", ...], start="2024-01-01")
    pipeline.update_incremental(codes)
    factors = pipeline.compute_all_factors(codes, end_date="2024-06-01")
    pipeline.feedback_from_performance(strategy_id, pnl=0.05, sharpe=1.2)
"""
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .news_handler import NewsHandler
from .announcement_handler import AnnouncementHandler
from .research_report_handler import ResearchReportHandler
from .macro_handler import MacroHandler

logger = logging.getLogger(__name__)

FEEDBACK_PATH = Path("data/alternative/feedback.json")


class AlternativeDataPipeline:
    """另类数据统一获取与因子计算管线"""

    def __init__(self, use_mock: bool = False):
        self.news = NewsHandler()
        self.announcement = AnnouncementHandler()
        self.report = ResearchReportHandler()
        self.macro = MacroHandler()
        self._use_mock = use_mock
        self._factor_weights: Dict[str, float] = {}
        self._feedback_history: List[Dict] = []
        self._load_feedback()

    def run_full_init(
        self,
        codes: List[str],
        start: str = "2024-01-01",
        end: str = None,
        include_macro: bool = True,
    ) -> Dict[str, int]:
        """一键全量初始化另类数据"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        results = {}

        if self._use_mock:
            from .mock_alternative import MockAlternativeGenerator
            gen = MockAlternativeGenerator()
            results = gen.generate_all(codes, start, end)
            return results

        logger.info(f"Initializing alternative data for {len(codes)} codes...")

        n_news = len(self.news.fetch_news(codes, start, end))
        results["news"] = n_news

        n_ann = len(self.announcement.fetch_announcements(codes, start, end))
        results["announcements"] = n_ann

        n_rating = len(self.report.fetch_ratings(codes))
        results["ratings"] = n_rating

        if include_macro:
            n_macro = len(self.macro.fetch_macro(start=start, end=end))
            results["macro"] = n_macro

        logger.info(f"Alternative data init complete: {results}")
        return results

    def update_incremental(self, codes: List[str]) -> Dict[str, int]:
        """增量更新另类数据（只拉取最新）"""
        today = datetime.now().strftime("%Y-%m-%d")
        results = {}

        if self._use_mock:
            from .mock_alternative import MockAlternativeGenerator
            gen = MockAlternativeGenerator()
            results = gen.generate_all(codes, start=today, end=today)
            return results

        results["news"] = len(self.news.fetch_news(codes, start=today, end=today))
        results["announcements"] = len(self.announcement.fetch_announcements(codes, start=today, end=today))
        results["ratings"] = len(self.report.fetch_ratings(codes, trade_date=today))

        macro_df = self.macro._load("macro_raw")
        if macro_df is not None and not macro_df.empty:
            last_date = macro_df.index[-1].strftime("%Y-%m-%d")
        else:
            last_date = "2020-01-01"
        results["macro"] = len(self.macro.fetch_macro(start=last_date, end=today))

        return results

    def compute_all_factors(
        self,
        codes: List[str],
        end_date: str,
        lookback: int = 20,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.Series]:
        """计算全部另类因子

        Returns:
            Dict[str, pd.Series]: 因子名 → 因子值Series(index=code)
        """
        factors = {}

        factors["news_sentiment"] = self.news.compute_sentiment_factor(
            codes, end_date, lookback
        )
        factors["news_attention"] = self.news.compute_attention_factor(
            codes, end_date, lookback
        )
        factors["announcement_event"] = self.announcement.compute_event_factor(
            codes, end_date, lookback
        )
        factors["dividend_indicator"] = self.announcement.compute_dividend_factor(
            codes, end_date, lookback=365
        )
        factors["consensus_rating"] = self.report.compute_consensus_factor(codes)
        factors["target_price_upside"] = self.report.compute_target_price_factor(
            codes, current_prices
        )
        factors["analyst_attention"] = self.report.compute_analyst_attention_factor(codes)

        macro_factors = self.macro.compute_macro_factor(end_date)
        for name, val in macro_factors.items():
            factors[name] = pd.Series(
                {c: val for c in codes}, name=name
            )

        return factors

    def verify_data(self, codes: List[str]) -> Dict[str, bool]:
        """验证各另类数据源是否就绪"""
        from pathlib import Path

        checks = {
            "news": bool(self.news._load("news_raw") is not None),
            "announcements": bool(self.announcement._load("announcements_raw") is not None),
            "ratings": bool(self.report._load("ratings_raw") is not None),
            "macro": bool(self.macro._load("macro_raw") is not None),
        }
        for name, ok in checks.items():
            status = "OK" if ok else "MISSING"
            logger.info(f"  {name}: {status}")
        return checks

    # ======== Data Feedback Interface ========

    def _load_feedback(self):
        """Load feedback history from disk."""
        if FEEDBACK_PATH.exists():
            try:
                data = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
                self._factor_weights = data.get("factor_weights", {})
                self._feedback_history = data.get("history", [])
                logger.info(
                    f"Feedback loaded: {len(self._feedback_history)} records, "
                    f"{len(self._factor_weights)} factor weights"
                )
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")

    def _save_feedback(self):
        """Persist feedback to disk."""
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEEDBACK_PATH.write_text(
            json.dumps(
                {
                    "factor_weights": self._factor_weights,
                    "history": self._feedback_history[-100:],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def feedback_from_performance(
        self,
        strategy_id: str,
        pnl: float,
        sharpe: float,
        max_drawdown: float = 0.0,
        factor_contributions: Optional[Dict[str, float]] = None,
    ):
        """Feed live performance back to adjust factor weights.

        When a strategy using alternative factors performs well/badly,
        the system adjusts the weights of those factors for future use.

        Args:
            strategy_id: Strategy identifier
            pnl: Period PnL (e.g. 0.05 for +5%)
            sharpe: Period Sharpe ratio
            max_drawdown: Period max drawdown
            factor_contributions: Per-factor contribution to returns
        """
        record = {
            "strategy_id": strategy_id,
            "timestamp": datetime.now().isoformat(),
            "pnl": pnl,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "factor_contributions": factor_contributions or {},
        }
        self._feedback_history.append(record)

        # Adjust factor weights based on performance
        if factor_contributions:
            for factor, contribution in factor_contributions.items():
                current_w = self._factor_weights.get(factor, 0.5)

                # Positive contribution → increase weight slightly
                # Negative contribution → decrease weight slightly
                adjustment = np.clip(contribution * 0.1, -0.05, 0.05)
                new_w = np.clip(current_w + adjustment, 0.1, 0.9)

                self._factor_weights[factor] = float(new_w)

        self._save_feedback()
        logger.info(
            f"Feedback recorded: {strategy_id} pnl={pnl:.4f} "
            f"sharpe={sharpe:.2f}"
        )

    def get_factor_weights(self) -> Dict[str, float]:
        """Get current factor weights (adjusted by feedback)."""
        return self._factor_weights.copy()

    def get_feedback_summary(self) -> Dict:
        """Get feedback history summary."""
        if not self._feedback_history:
            return {"total_records": 0}

        pnls = [r["pnl"] for r in self._feedback_history]
        sharpes = [r["sharpe"] for r in self._feedback_history]

        return {
            "total_records": len(self._feedback_history),
            "avg_pnl": float(np.mean(pnls)),
            "avg_sharpe": float(np.mean(sharpes)),
            "best_pnl": max(pnls),
            "worst_pnl": min(pnls),
            "factor_weights": self._factor_weights,
        }

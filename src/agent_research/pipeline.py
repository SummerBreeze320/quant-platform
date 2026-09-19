from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session

from src.common.logger import logger
from src.models.factor import FactorMetadata
from src.agent_research.evaluator import FactorEvaluator, FactorEvaluationMetrics
from src.agent_research.hypothesis import FactorHypothesis, FactorHypothesisEngine


class AutonomousFactorPipeline:
    """
    LLM 驱动的自主量化投研因子挖掘与生命周期管理流水线：
    1. 生成金融逻辑假设与表达式；
    2. 计算行情特征矩阵与截面因子值；
    3. 全面统计评估 (IC / Rank IC / ICIR / t-stat / 分层单调性 / 换手)；
    4. 严格执行质量门禁 (Quality Gate)，合格因子自动入库；
    5. 定期回测衰减检测，自动淘汰失效退化因子。
    """

    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        # 内存回退注册表 (当数据库脱机或轻量化无持久化运行时)
        self._in_memory_registry: Dict[str, Dict[str, Any]] = {}

    def run_mining_cycle(
        self,
        price_data: Dict[str, pd.DataFrame],
        category: Optional[str] = None,
        count: int = 3,
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """执行单次自主投研挖掘与考核循环"""
        close_df = price_data.get("close", pd.DataFrame())
        if close_df.empty or len(close_df) < 10:
            logger.warning("[FactorPipeline] Insufficient price data for mining.")
            return []

        # 1. 产生前瞻 1 期收益率
        forward_rets = close_df.pct_change(1).shift(-1)

        # 2. 启发式或 LLM 生成候选假设
        hypotheses = FactorHypothesisEngine.generate_hypotheses(
            category=category,
            count=count,
            use_llm=True,
        )

        results: List[Dict[str, Any]] = []

        for hyp in hypotheses:
            logger.info(f"[FactorPipeline] Evaluating candidate factor '{hyp.name}' ({hyp.category})...")
            # 3. 计算因子时间序列值
            factor_df = FactorHypothesisEngine.compute_factor_values(hyp, price_data)

            # 4. 评估全套统计指标
            metrics = FactorEvaluator.evaluate(
                factor_df=factor_df,
                forward_returns_df=forward_rets,
                ic_threshold=ic_threshold,
                icir_threshold=icir_threshold,
            )

            # 5. 准入入库或淘汰归档
            record = self._register_or_update_factor(hyp, metrics)
            results.append({
                "name": hyp.name,
                "category": hyp.category,
                "expression": hyp.expression,
                "metrics": metrics.model_dump(),
                "is_active": metrics.passed_gate,
                "record_id": getattr(record, "id", None),
            })

            status_str = "PASSED & ACTIVATED" if metrics.passed_gate else "REJECTED"
            logger.info(f"[FactorPipeline] Factor '{hyp.name}': {status_str} (IC: {metrics.ic_mean}, ICIR: {metrics.icir})")

        return results

    def retire_decayed_factors(
        self,
        factor_evaluations: Dict[str, FactorEvaluationMetrics],
        min_icir: float = 0.3,
    ) -> List[str]:
        """
        因子衰减生命周期管理：
        若某存量活跃因子的最新 ICIR 衰减跌破最低存续阈值，则自动置为 is_active = False 淘汰下架。
        """
        retired_names: List[str] = []

        # 1. 数据库模式淘汰处理
        if self.db is not None:
            active_factors = self.db.query(FactorMetadata).filter_by(is_active=True).all()
            for fac in active_factors:
                if fac.name in factor_evaluations:
                    eval_res = factor_evaluations[fac.name]
                    if abs(eval_res.icir) < min_icir or abs(eval_res.rank_icir) < min_icir:
                        fac.is_active = False
                        fac.extra_metrics = fac.extra_metrics or {}
                        fac.extra_metrics["deprecation_reason"] = f"ICIR decayed to {eval_res.icir:.2f} (threshold: {min_icir})"
                        fac.extra_metrics["retired_at"] = datetime.now().isoformat()
                        retired_names.append(fac.name)
            if retired_names:
                self.db.commit()

        # 2. 内存注册表同步更新
        for name, entry in self._in_memory_registry.items():
            if entry.get("is_active") and name in factor_evaluations:
                eval_res = factor_evaluations[name]
                if abs(eval_res.icir) < min_icir:
                    entry["is_active"] = False
                    entry["deprecation_reason"] = f"ICIR decayed to {eval_res.icir:.2f}"
                    if name not in retired_names:
                        retired_names.append(name)

        if retired_names:
            logger.info(f"[FactorPipeline] Retired {len(retired_names)} decayed factors: {retired_names}")
        return retired_names

    def list_active_factors(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """查询当前生效准入的全部因子"""
        if self.db is not None:
            query = self.db.query(FactorMetadata).filter_by(is_active=True)
            if category:
                query = query.filter_by(category=category)
            return [
                {
                    "name": f.name,
                    "expression": f.expression,
                    "category": f.category,
                    "ic_mean": f.ic_mean,
                    "icir": f.icir,
                    "created_by": f.created_by,
                }
                for f in query.all()
            ]

        # 内存回退
        active = [v for v in self._in_memory_registry.values() if v.get("is_active")]
        if category:
            active = [v for v in active if v.get("category") == category]
        return active

    def _register_or_update_factor(
        self,
        hyp: FactorHypothesis,
        metrics: FactorEvaluationMetrics,
    ) -> Any:
        """登记或更新因子记录"""
        is_active = metrics.passed_gate

        if self.db is not None:
            existing = self.db.query(FactorMetadata).filter_by(name=hyp.name).first()
            if existing:
                existing.expression = hyp.expression
                existing.category = hyp.category
                existing.description = hyp.hypothesis
                existing.ic_mean = metrics.ic_mean
                existing.icir = metrics.icir
                existing.t_stat = metrics.t_stat
                existing.extra_metrics = metrics.model_dump()
                existing.is_active = is_active
                self.db.commit()
                self.db.refresh(existing)
                return existing

            new_record = FactorMetadata(
                name=hyp.name,
                expression=hyp.expression,
                category=hyp.category,
                description=hyp.hypothesis,
                ic_mean=metrics.ic_mean,
                icir=metrics.icir,
                t_stat=metrics.t_stat,
                extra_metrics=metrics.model_dump(),
                created_by="AutonomousAgent",
                is_active=is_active,
            )
            self.db.add(new_record)
            self.db.commit()
            self.db.refresh(new_record)
            return new_record

        # 内存注册
        entry = {
            "name": hyp.name,
            "expression": hyp.expression,
            "category": hyp.category,
            "description": hyp.hypothesis,
            "ic_mean": metrics.ic_mean,
            "icir": metrics.icir,
            "t_stat": metrics.t_stat,
            "extra_metrics": metrics.model_dump(),
            "created_by": "AutonomousAgent",
            "is_active": is_active,
            "updated_at": datetime.now().isoformat(),
        }
        self._in_memory_registry[hyp.name] = entry
        return entry

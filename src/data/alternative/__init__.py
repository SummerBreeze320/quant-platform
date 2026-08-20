"""另类数据接入层

新闻舆情、公告事件、研报评级、宏观经济四类另类数据的
获取、存储、因子计算统一入口。

对齐七层架构中的Data Layer扩展，为SignalGenerator提供另类因子。
"""
from .news_handler import NewsHandler
from .announcement_handler import AnnouncementHandler
from .research_report_handler import ResearchReportHandler
from .macro_handler import MacroHandler, MACRO_INDICATORS
from .alternative_pipeline import AlternativeDataPipeline
from .mock_alternative import MockAlternativeGenerator

__all__ = [
    "NewsHandler",
    "AnnouncementHandler",
    "ResearchReportHandler",
    "MacroHandler",
    "MACRO_INDICATORS",
    "AlternativeDataPipeline",
    "MockAlternativeGenerator",
]

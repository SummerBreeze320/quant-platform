"""Qlib instrument file generator.

Format: {code}\t{start_date}\t{end_date}
e.g.: SH600000\t2015-01-05\t2026-08-01
"""
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class InstrumentGenerator:
    """Generate Qlib instrument files."""

    def __init__(self, qlib_dir: str = "data/qlib_bin"):
        self.qlib_dir = Path(qlib_dir)
        self.instruments_dir = self.qlib_dir / "instruments"
        self.instruments_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        name: str,
        codes: List[str],
        start_dates: Optional[List[str]] = None,
        end_dates: Optional[List[str]] = None,
    ) -> Path:
        """Generate instrument file.

        Args:
            name: File name without extension, e.g. "all", "csi300", "etf"
            codes: Instrument codes like ["SH600000", "SZ000001"]
            start_dates: Per-instrument listing dates (same length as codes)
            end_dates: Per-instrument end dates, or a single end date string

        Returns:
            Path to the generated file.
        """
        path = self.instruments_dir / f"{name}.txt"
        default_start = "2015-01-01"
        default_end = "2026-12-31"

        # Normalize end_dates to a list
        if end_dates is None:
            end_list = [default_end] * len(codes)
        elif isinstance(end_dates, str):
            end_list = [end_dates] * len(codes)
        else:
            end_list = list(end_dates)
            if len(end_list) < len(codes):
                end_list.extend([default_end] * (len(codes) - len(end_list)))

        lines = []
        for i, code in enumerate(codes):
            start = start_dates[i] if start_dates and i < len(start_dates) else default_start
            end = end_list[i] if i < len(end_list) else default_end
            lines.append(f"{code}\t{start}\t{end}")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated instrument file: {name}.txt ({len(codes)} instruments)")
        return path

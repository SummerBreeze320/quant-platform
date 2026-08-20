"""Feature Store — 持久化特征存储

Provides versioned, cross-experiment feature reuse with:
- Versioned factor storage (parquet + metadata)
- Feature serving API (get/save/query)
- Cross-experiment reuse (avoid recomputation)
- Incremental updates (append new dates only)
- IC tracking per feature version

Storage layout:
    data/feature_store/
        v1/
            kmid.parquet
            std_20.parquet
            metadata.json
        v2/
            ...
"""
import logging
import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

STORE_ROOT = Path("data/feature_store")


class FeatureVersion:
    """Metadata for one feature version."""

    def __init__(
        self,
        name: str,
        version: str,
        expression: str,
        instruments: List[str],
        date_range: Tuple[str, str],
        created_at: str,
        ic_score: float = 0.0,
        icir_score: float = 0.0,
        file_path: str = "",
    ):
        self.name = name
        self.version = version
        self.expression = expression
        self.instruments = instruments
        self.date_range = date_range
        self.created_at = created_at
        self.ic_score = ic_score
        self.icir_score = icir_score
        self.file_path = file_path

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "expression": self.expression,
            "instruments": self.instruments,
            "date_range": list(self.date_range),
            "created_at": self.created_at,
            "ic_score": self.ic_score,
            "icir_score": self.icir_score,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureVersion":
        return cls(
            name=d["name"],
            version=d["version"],
            expression=d["expression"],
            instruments=d["instruments"],
            date_range=tuple(d["date_range"]),
            created_at=d["created_at"],
            ic_score=d.get("ic_score", 0),
            icir_score=d.get("icir_score", 0),
            file_path=d.get("file_path", ""),
        )


class FeatureStore:
    """Persistent feature store with versioning.

    Features:
    - save: store computed factors with version tag
    - get: retrieve factors by name + version
    - get_latest: get most recent version
    - query: find features matching criteria
    - update_ic: record IC/ICIR for a feature version
    - list_versions: all versions of a feature
    - prune: remove old versions keeping only N most recent
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or STORE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "metadata.json"
        self._metadata: Dict[str, List[Dict]] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata index from JSON."""
        if self._meta_path.exists():
            try:
                self._metadata = json.loads(
                    self._meta_path.read_text(encoding="utf-8")
                )
                logger.info(
                    f"FeatureStore loaded: "
                    f"{sum(len(v) for v in self._metadata.values())} versions"
                )
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self._metadata = {}

    def _save_metadata(self):
        """Persist metadata index."""
        self._meta_path.write_text(
            json.dumps(self._metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _version_dir(self, version: str) -> Path:
        """Get directory for a version."""
        d = self.root / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _feature_key(
        self,
        name: str,
        instruments: List[str],
        date_range: Tuple[str, str],
    ) -> str:
        """Generate a hash key for feature identity."""
        key_str = f"{name}|{'-'.join(sorted(instruments))}|{date_range[0]}|{date_range[1]}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def save(
        self,
        name: str,
        expression: str,
        data: pd.DataFrame,
        instruments: List[str],
        date_range: Tuple[str, str],
        version: Optional[str] = None,
        ic_score: float = 0.0,
        icir_score: float = 0.0,
    ) -> str:
        """Save a computed feature to the store.

        Args:
            name: Feature name (e.g. "kmid", "std_20")
            expression: Qlib expression or description
            data: DataFrame with MultiIndex (instrument, datetime)
            instruments: List of instrument codes
            date_range: (start_date, end_date)
            version: Version tag. Auto-generated if None.
            ic_score: IC score for this feature
            icir_score: ICIR score

        Returns:
            Version string
        """
        if version is None:
            existing = self._metadata.get(name, [])
            version = f"v{len(existing) + 1}"

        file_name = f"{name}_{version}.parquet"
        file_path = self._version_dir(version) / file_name

        data.to_parquet(file_path, engine="pyarrow")

        fv = FeatureVersion(
            name=name,
            version=version,
            expression=expression,
            instruments=instruments,
            date_range=date_range,
            created_at=datetime.now().isoformat(),
            ic_score=round(ic_score, 6),
            icir_score=round(icir_score, 6),
            file_path=str(file_path),
        )

        if name not in self._metadata:
            self._metadata[name] = []
        self._metadata[name].append(fv.to_dict())
        self._save_metadata()

        logger.info(
            f"FeatureStore saved: {name} {version} "
            f"({len(data)} rows, {len(instruments)} instruments, "
            f"IC={ic_score:.4f})"
        )
        return version

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve a feature from the store.

        Args:
            name: Feature name
            version: Version tag. None = latest.

        Returns:
            DataFrame or None if not found.
        """
        versions = self._metadata.get(name, [])
        if not versions:
            return None

        if version is None:
            target = versions[-1]
        else:
            target = next((v for v in versions if v["version"] == version), None)
            if target is None:
                logger.warning(f"Version {version} not found for {name}")
                return None

        file_path = Path(target["file_path"])
        if not file_path.exists():
            logger.warning(f"Feature file missing: {file_path}")
            return None

        return pd.read_parquet(file_path)

    def get_latest(self, name: str) -> Optional[pd.DataFrame]:
        """Get the latest version of a feature."""
        return self.get(name, version=None)

    def query(
        self,
        name: Optional[str] = None,
        min_ic: float = 0.0,
        min_icir: float = 0.0,
        instruments_subset: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Query features matching criteria.

        Args:
            name: Filter by feature name
            min_ic: Minimum IC score
            min_icir: Minimum ICIR score
            instruments_subset: Must include at least one of these instruments

        Returns:
            List of matching feature version metadata dicts
        """
        results = []

        names = [name] if name else list(self._metadata.keys())
        for n in names:
            for v in self._metadata.get(n, []):
                if v.get("ic_score", 0) < min_ic:
                    continue
                if v.get("icir_score", 0) < min_icir:
                    continue
                if instruments_subset:
                    stored = set(v.get("instruments", []))
                    if not any(i in stored for i in instruments_subset):
                        continue
                results.append(v)

        return results

    def update_ic(
        self,
        name: str,
        version: str,
        ic_score: float,
        icir_score: float,
    ):
        """Update IC/ICIR scores for a feature version."""
        versions = self._metadata.get(name, [])
        for v in versions:
            if v["version"] == version:
                v["ic_score"] = round(ic_score, 6)
                v["icir_score"] = round(icir_score, 6)
                self._save_metadata()
                logger.info(f"Updated IC for {name} {version}: IC={ic_score:.4f}")
                return
        logger.warning(f"Version {version} not found for {name}")

    def list_versions(self, name: str) -> List[Dict]:
        """List all versions of a feature."""
        return self._metadata.get(name, [])

    def list_features(self) -> List[str]:
        """List all feature names in the store."""
        return list(self._metadata.keys())

    def prune(self, keep: int = 3):
        """Remove old versions, keeping only the N most recent per feature."""
        for name, versions in self._metadata.items():
            if len(versions) <= keep:
                continue

            to_remove = versions[:-keep]
            for v in to_remove:
                file_path = Path(v["file_path"])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Pruned {name} {v['version']}")

            self._metadata[name] = versions[-keep:]

        self._save_metadata()

    def append_incremental(
        self,
        name: str,
        expression: str,
        new_data: pd.DataFrame,
        instruments: List[str],
        date_range: Tuple[str, str],
    ) -> str:
        """Append new data to the latest version (incremental update).

        If the feature exists, append new rows. Otherwise, create v1.
        """
        latest = self.get_latest(name)

        if latest is not None and not latest.empty:
            combined = pd.concat([latest, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()

            versions = self._metadata.get(name, [])
            version = f"v{len(versions) + 1}"
            return self.save(
                name=name,
                expression=expression,
                data=combined,
                instruments=instruments,
                date_range=date_range,
                version=version,
            )
        else:
            return self.save(
                name=name,
                expression=expression,
                data=new_data,
                instruments=instruments,
                date_range=date_range,
            )

    def summary(self) -> Dict[str, Any]:
        """Get store summary statistics."""
        total_versions = sum(len(v) for v in self._metadata.values())
        features = list(self._metadata.keys())

        ic_scores = []
        for versions in self._metadata.values():
            for v in versions:
                if v.get("ic_score", 0) != 0:
                    ic_scores.append(v["ic_score"])

        return {
            "total_features": len(features),
            "total_versions": total_versions,
            "features": features,
            "avg_ic": float(np.mean(ic_scores)) if ic_scores else 0,
            "best_ic": max(ic_scores) if ic_scores else 0,
        }

    def serve(
        self,
        names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Serve multiple features as a merged DataFrame.

        This is the feature serving API for Agents and models.

        Args:
            names: Feature names to serve
            start_date/end_date: Optional date filter

        Returns:
            Merged DataFrame with all requested features
        """
        frames = []
        valid_names = []

        for name in names:
            df = self.get_latest(name)
            if df is not None and not df.empty:
                if start_date:
                    df = df[df.index.get_level_values("datetime") >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index.get_level_values("datetime") <= pd.Timestamp(end_date)]
                frames.append(df)
                valid_names.append(name)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, axis=1)
        merged.columns = valid_names
        return merged

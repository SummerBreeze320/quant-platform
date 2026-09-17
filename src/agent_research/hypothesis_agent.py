import json
import random
from typing import Dict, Any, Optional
from src.common.config import get_settings
from src.common.logger import logger
from src.agent_research.prompt_templates import HYPOTHESIS_SYSTEM_PROMPT

class HypothesisAgent:
    """Proposes economically motivated quantitative factor hypotheses."""

    HEURISTIC_HYPOTHESES = [
        {
            "hypothesis_name": "short_term_reversal_5d",
            "category": "reversal",
            "description": "Stocks with extreme positive returns over the past 5 trading days tend to mean-revert due to liquidity provision and retail overshooting.",
            "mathematical_concept": "Negative 5-day return scaled by 20-day volatility"
        },
        {
            "hypothesis_name": "volume_surge_price_drift",
            "category": "volume_price",
            "description": "Abnormal trading volume surges accompanied by modest price gains indicate institutional accumulation and predict post-announcement drift.",
            "mathematical_concept": "Volume relative to 20d moving average times intraday return"
        },
        {
            "hypothesis_name": "momentum_volatility_adjusted_20d",
            "category": "momentum",
            "description": "Price momentum over 20 days penalized by intraday volatility yields higher risk-adjusted information coefficient.",
            "mathematical_concept": "20d cumulative return divided by 20d standard deviation of daily returns"
        },
        {
            "hypothesis_name": "vwap_close_divergence",
            "category": "volume_price",
            "description": "Closing prices closing significantly above the daily volume-weighted average price (VWAP) signify late-day buying pressure.",
            "mathematical_concept": "($close - $vwap) / ($vwap + 1e-5)"
        }
    ]

    def __init__(self, llm_client: Optional[Any] = None):
        self.settings = get_settings()
        self.llm_client = llm_client

    def propose_hypothesis(
        self,
        theme: str = "reversal",
        market_context: str = "A-Share CSI 300 / CSI 500"
    ) -> Dict[str, Any]:
        """Generates a quantitative hypothesis for factor exploration."""
        if self.llm_client is not None:
            try:
                # LLM execution
                prompt = f"Theme: {theme}\nMarket Context: {market_context}\nGenerate a hypothesis."
                response = self.llm_client.chat.completions.create(
                    model=self.settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                # Parse json from content
                cleaned = content.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
                return json.loads(cleaned)
            except Exception as e:
                logger.warning(f"LLM hypothesis generation failed, falling back to heuristics: {e}")

        # Fallback heuristic selection
        matching = [h for h in self.HEURISTIC_HYPOTHESES if h["category"] == theme]
        if matching:
            return random.choice(matching)
        return random.choice(self.HEURISTIC_HYPOTHESES)

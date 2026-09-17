import re
from typing import Optional, Any, Dict
from src.common.config import get_settings
from src.common.logger import logger
from src.agent_research.prompt_templates import FACTOR_CODER_PROMPT

class FactorCoderAgent:
    """Translates financial hypotheses into valid Qlib Alpha expressions."""

    HEURISTIC_EXPRESSIONS = {
        "short_term_reversal_5d": "-1 * (Ref($close, -1) - Ref($close, -5)) / (Ref($close, -5) + 1e-6)",
        "volume_surge_price_drift": "($close - $open) / ($open + 1e-6) * ($volume / (Mean($volume, 20) + 1e-6))",
        "momentum_volatility_adjusted_20d": "(Ref($close, -1) - Ref($close, -20)) / (Std($close / Ref($close, -1) - 1, 20) + 1e-6)",
        "vwap_close_divergence": "($close - $vwap) / ($vwap + 1e-6)"
    }

    def __init__(self, llm_client: Optional[Any] = None):
        self.settings = get_settings()
        self.llm_client = llm_client

    @staticmethod
    def clean_expression(raw_text: str) -> str:
        """Strips markdown markers, comments and whitespace from formula output."""
        text = raw_text.strip()
        # Remove ```qlib ... ``` or ```python ... ```
        if "```" in text:
            match = re.search(r"```(?:\w+)?\n?(.*?)\n?```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                text = text.replace("```", "").strip()
        # Remove any leading 'formula = '
        if "=" in text and not any(op in text.split("=")[0] for op in ["<", ">", "!", "+", "-", "*", "/"]):
            text = text.split("=", 1)[1].strip()
        return text.strip()

    def generate_code(self, hypothesis: Dict[str, Any]) -> str:
        """Generates Qlib factor expression string for given hypothesis."""
        hyp_name = hypothesis.get("hypothesis_name", "")

        if self.llm_client is not None:
            try:
                user_msg = f"Hypothesis Name: {hyp_name}\nCategory: {hypothesis.get('category')}\nDescription: {hypothesis.get('description')}\nMathematical Concept: {hypothesis.get('mathematical_concept')}\nGenerate Qlib expression."
                response = self.llm_client.chat.completions.create(
                    model=self.settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": FACTOR_CODER_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.2
                )
                raw_code = response.choices[0].message.content
                cleaned = self.clean_expression(raw_code)
                if cleaned:
                    return cleaned
            except Exception as e:
                logger.warning(f"LLM factor code generation failed: {e}")

        # Fallback heuristic
        if hyp_name in self.HEURISTIC_EXPRESSIONS:
            return self.HEURISTIC_EXPRESSIONS[hyp_name]
        
        # Generic safe expression
        return "($close - Mean($close, 10)) / (Std($close, 10) + 1e-6)"

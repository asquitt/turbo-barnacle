"""
LLM Hypothesis Generation Agent

This agent acts as a "principal investigator" in the materials discovery loop:
- Analyzes current discovery progress
- Proposes next search spaces based on domain knowledge
- Suggests acquisition functions
- Critiques predictions for quality control

The agent uses Claude 3.5 Sonnet for reasoning and domain expertise.

Key Design Principles:
1. Cost efficiency: Cache responses, compress prompts, use appropriate model sizes
2. Scientific rigor: Ground reasoning in materials science principles
3. Interpretability: Structured outputs with reasoning traces
4. Safety: Validate outputs before use

Author: Materials Discovery Team
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class SearchSpaceProposal:
    """
    Structured output from hypothesis agent.

    Attributes:
        elements: List of elements to explore
        num_elements_range: (min, max) elements per composition
        structure_prototypes: Crystal structure templates to try
        reasoning: Agent's reasoning for this proposal
        confidence: Agent's confidence in this hypothesis (0-1)
        priority: Priority level (1=highest)
        estimated_success_rate: Estimated discovery rate
    """
    elements: List[str]
    num_elements_range: Tuple[int, int]
    structure_prototypes: List[str]
    reasoning: str
    confidence: float
    priority: int = 1
    estimated_success_rate: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CritiqueResult:
    """
    Result from prediction critique.

    Attributes:
        flagged: Whether prediction is suspicious
        reasons: List of concerns
        suggestions: Suggestions for improvement
        confidence_adjustment: Suggested confidence adjustment
    """
    flagged: bool
    reasons: List[str]
    suggestions: List[str]
    confidence_adjustment: float = 0.0


class HypothesisAgent:
    """
    LLM agent for guiding materials discovery.

    This agent provides domain expertise and strategic reasoning to:
    1. Propose promising search spaces
    2. Interpret discovery results
    3. Adapt strategy based on progress
    4. Critique model predictions

    Cost Control Features:
    - Response caching (semantic similarity)
    - Token counting and budgets
    - Prompt compression
    - Model selection (Sonnet vs Haiku)

    Example:
        >>> agent = HypothesisAgent(api_key="your_key")
        >>> proposal = agent.propose_search_space(
        ...     current_best=[...],
        ...     previous_attempts=[...],
        ...     target_property="low_formation_energy"
        ... )
        >>> print(f"Try: {proposal.elements}")
        >>> print(f"Reasoning: {proposal.reasoning}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        fallback_model: str = "claude-3-haiku-20240307",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        cache_dir: str = "data/cache/agent_responses",
        use_caching: bool = True,
        max_cost_per_experiment: float = 5.0,
    ):
        """
        Initialize the hypothesis agent.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Primary model to use
            fallback_model: Cheaper model for simple tasks
            temperature: Sampling temperature (0=deterministic, 1=creative)
            max_tokens: Maximum response length
            cache_dir: Directory for caching responses
            use_caching: Enable response caching
            max_cost_per_experiment: Maximum USD to spend per experiment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_caching = use_caching

        # Cost tracking
        self.max_cost_per_experiment = max_cost_per_experiment
        self.total_cost = 0.0
        self.total_tokens = {'input': 0, 'output': 0}

        # Token counter (for cost estimation)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # Close enough
        except:
            self.tokenizer = None
            logger.warning("Token counter not available")

        logger.info(f"Initialized HypothesisAgent with {model}")

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.use_caching:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                logger.debug(f"Cache hit: {cache_key}")
                return data['response']

        return None

    def _cache_response(self, cache_key: str, response: str, metadata: Dict):
        """Cache response to disk."""
        if not self.use_caching:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'response': response,
                'metadata': metadata,
            }, f, indent=2)

        logger.debug(f"Cached response: {cache_key}")

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate API cost.

        Claude 3.5 Sonnet pricing (as of 2024):
        - Input: $3 per 1M tokens
        - Output: $15 per 1M tokens

        Claude 3 Haiku pricing:
        - Input: $0.25 per 1M tokens
        - Output: $1.25 per 1M tokens
        """
        if "sonnet" in self.model.lower():
            cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0
        elif "haiku" in self.model.lower():
            cost = (input_tokens / 1_000_000) * 0.25 + (output_tokens / 1_000_000) * 1.25
        else:
            # Conservative estimate
            cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0

        return cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_fallback: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Call Claude API with retry logic.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            use_fallback: Use cheaper Haiku model

        Returns:
            (response_text, metadata)
        """
        # Check cost budget
        if self.total_cost >= self.max_cost_per_experiment:
            raise RuntimeError(
                f"Cost budget exceeded: ${self.total_cost:.2f} / ${self.max_cost_per_experiment:.2f}"
            )

        # Check cache
        cache_key = self._cache_key(prompt + (system_prompt or ""))
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached, {'cached': True}

        # Select model
        model = self.fallback_model if use_fallback else self.model

        # Call API
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt if system_prompt else "",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response
            response_text = response.content[0].text

            # Track usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._estimate_cost(input_tokens, output_tokens)

            self.total_tokens['input'] += input_tokens
            self.total_tokens['output'] += output_tokens
            self.total_cost += cost

            metadata = {
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost,
                'total_cost': self.total_cost,
                'cached': False,
            }

            logger.info(
                f"API call: {input_tokens} in + {output_tokens} out = ${cost:.4f}"
            )

            # Cache response
            self._cache_response(cache_key, response_text, metadata)

            return response_text, metadata

        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def propose_search_space(
        self,
        current_best: List[Dict],
        previous_attempts: List[SearchSpaceProposal],
        target_property: str = "low_formation_energy",
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> SearchSpaceProposal:
        """
        Propose next search space to explore.

        The agent analyzes:
        - Current best materials (what's working?)
        - Previous search spaces (what didn't work?)
        - Progress vs. budget (explore or exploit?)
        - Domain knowledge (chemical intuition)

        Args:
            current_best: Top materials found so far
            previous_attempts: Previous search space proposals
            target_property: Optimization objective
            iteration: Current iteration number
            total_iterations: Total budget

        Returns:
            SearchSpaceProposal with next search strategy
        """
        # Build prompt
        system_prompt = """You are an expert materials scientist specializing in inorganic crystals and computational materials discovery. Your task is to propose promising search spaces for discovering new materials with desired properties.

Use your knowledge of:
- Crystal structure stability (Hume-Rothery rules, Goldschmidt tolerance factor)
- Electronic structure (band gap engineering, d-orbital splitting)
- Chemical intuition (electronegativity, ionic radii, oxidation states)
- Materials databases (common structure types, known stable phases)

Propose search spaces that balance exploration (finding new chemical spaces) and exploitation (refining promising leads)."""

        # Format current best materials
        best_materials_str = "\n".join([
            f"  - {m.get('formula', 'Unknown')}: "
            f"E_form={m.get('formation_energy', 'N/A'):.3f} eV/atom, "
            f"E_gap={m.get('band_gap', 'N/A'):.2f} eV"
            for m in current_best[:5]  # Top 5
        ])

        # Format previous attempts
        attempts_str = "\n".join([
            f"  {i+1}. Elements: {', '.join(attempt.elements)}, "
            f"Structures: {', '.join(attempt.structure_prototypes)}, "
            f"Confidence: {attempt.confidence:.2f}"
            for i, attempt in enumerate(previous_attempts[-3:])  # Last 3
        ])

        # Exploration vs exploitation guidance
        progress = iteration / total_iterations
        if progress < 0.3:
            stage = "early exploration - prioritize diversity"
        elif progress < 0.7:
            stage = "exploitation - refine promising leads"
        else:
            stage = "final optimization - focus on best regions"

        user_prompt = f"""Current Discovery Progress:

**Target**: {target_property}
**Iteration**: {iteration} / {total_iterations} ({stage})

**Current Best Materials**:
{best_materials_str if best_materials_str else "  (none yet)"}

**Previous Search Spaces** (last 3):
{attempts_str if attempts_str else "  (none yet)"}

Based on this information, propose the next search space to explore. Consider:
1. What chemical patterns are emerging in successful materials?
2. What unexplored regions might be promising?
3. Should we explore (new chemistry) or exploit (refine current)?
4. What structure types are most likely to succeed?

Provide your response in the following JSON format:
{{
  "elements": ["Element1", "Element2", ...],
  "num_elements_range": [min, max],
  "structure_prototypes": ["prototype1", "prototype2", ...],
  "reasoning": "Your detailed reasoning...",
  "confidence": 0.0-1.0,
  "priority": 1-5,
  "estimated_success_rate": 0.0-1.0
}}

Available structure prototypes:
- perovskite (ABX3): Good for oxides with large cation size mismatch
- spinel (AB2X4): Stable for mixed valence transition metals
- rocksalt (AB): Simple binary compounds
- wurtzite (AB): Covalent bonding, wide band gaps
- fluorite (AB2): High coordination, dense packing
- pyrochlore (A2B2X7): Complex oxides, frustrated magnetism

Be specific about why you chose these elements and structures."""

        # Call API (use fallback for simple cases)
        use_fallback = iteration < 5 and len(current_best) == 0  # Early random exploration
        response, metadata = self._call_api(user_prompt, system_prompt, use_fallback)

        # Parse JSON response
        try:
            # Extract JSON from response (may have extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]

            data = json.loads(json_str)

            proposal = SearchSpaceProposal(
                elements=data['elements'],
                num_elements_range=tuple(data['num_elements_range']),
                structure_prototypes=data['structure_prototypes'],
                reasoning=data['reasoning'],
                confidence=data['confidence'],
                priority=data.get('priority', 1),
                estimated_success_rate=data.get('estimated_success_rate'),
            )

            logger.info(f"Proposed search space: {proposal.elements}")
            return proposal

        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            logger.debug(f"Response: {response}")

            # Fallback to simple random proposal
            return SearchSpaceProposal(
                elements=["Fe", "O"],
                num_elements_range=(2, 3),
                structure_prototypes=["rocksalt", "perovskite"],
                reasoning="Fallback proposal due to parsing error",
                confidence=0.3,
                priority=3,
            )

    def critique_prediction(
        self,
        formula: str,
        structure_description: str,
        predicted_energy: float,
        predicted_bandgap: float,
        predicted_stability: float,
        uncertainty: Optional[Dict[str, float]] = None,
    ) -> CritiqueResult:
        """
        Critique a model prediction for quality control.

        The agent checks:
        - Physical plausibility
        - Consistency with known chemistry
        - Uncertainty levels
        - Potential modeling errors

        Args:
            formula: Chemical formula
            structure_description: Structure type and details
            predicted_energy: Formation energy (eV/atom)
            predicted_bandgap: Band gap (eV)
            predicted_stability: Stability probability
            uncertainty: Uncertainty estimates (optional)

        Returns:
            CritiqueResult with concerns and suggestions
        """
        system_prompt = """You are a critical reviewer of computational materials predictions. Your job is to identify potentially unreliable predictions and suggest improvements."""

        user_prompt = f"""Review this prediction:

**Material**: {formula}
**Structure**: {structure_description}
**Predicted Formation Energy**: {predicted_energy:.3f} eV/atom
**Predicted Band Gap**: {predicted_bandgap:.2f} eV
**Predicted Stability**: {predicted_stability:.2f}

"""

        if uncertainty:
            user_prompt += f"""**Uncertainties**:
  - Energy: ± {uncertainty.get('energy', 0):.3f} eV/atom
  - Band gap: ± {uncertainty.get('bandgap', 0):.2f} eV
  - Stability: ± {uncertainty.get('stability', 0):.2f}

"""

        user_prompt += """Check for:
1. Unrealistic property values
2. High uncertainty
3. Unusual chemical combinations
4. Known physics violations

Response format:
{
  "flagged": true/false,
  "reasons": ["reason1", "reason2", ...],
  "suggestions": ["suggestion1", "suggestion2", ...],
  "confidence_adjustment": -0.5 to +0.5
}"""

        # Use cheap Haiku model for critique
        response, _ = self._call_api(user_prompt, system_prompt, use_fallback=True)

        # Parse response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            data = json.loads(response[json_start:json_end])

            return CritiqueResult(
                flagged=data['flagged'],
                reasons=data['reasons'],
                suggestions=data['suggestions'],
                confidence_adjustment=data.get('confidence_adjustment', 0.0),
            )

        except:
            # Conservative fallback: don't flag
            return CritiqueResult(
                flagged=False,
                reasons=[],
                suggestions=["Could not parse critique"],
            )

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost and usage summary."""
        return {
            'total_cost': self.total_cost,
            'max_budget': self.max_cost_per_experiment,
            'budget_remaining': self.max_cost_per_experiment - self.total_cost,
            'input_tokens': self.total_tokens['input'],
            'output_tokens': self.total_tokens['output'],
            'total_tokens': sum(self.total_tokens.values()),
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Note: Requires ANTHROPIC_API_KEY environment variable
    # agent = HypothesisAgent()

    print("HypothesisAgent module loaded successfully")
    print()
    print("To use:")
    print("  1. Set ANTHROPIC_API_KEY environment variable")
    print("  2. agent = HypothesisAgent()")
    print("  3. proposal = agent.propose_search_space(...)")

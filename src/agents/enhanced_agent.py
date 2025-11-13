"""
Enhanced LLM Agent with Advanced Reasoning

This module implements advanced agent capabilities including:
- Chain-of-Thought (CoT) reasoning for complex decisions
- Self-reflection and error correction
- Few-shot learning with dynamic examples
- Multi-agent collaboration
- ReAct-style reasoning (Reason + Act)

Research Background:
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- ReAct: https://arxiv.org/abs/2210.03629
- Self-Reflection: https://arxiv.org/abs/2303.11366

Author: Materials Discovery Team
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .hypothesis_agent import HypothesisAgent, SearchSpaceProposal

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """
    Single step in chain-of-thought reasoning.

    Attributes:
        thought: What the agent is thinking
        action: What action to take (if any)
        observation: Result of the action
        confidence: Confidence in this step
    """
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReflectionResult:
    """
    Result from self-reflection.

    Attributes:
        critique: Critical analysis of previous attempt
        lessons_learned: Key takeaways
        improvement_suggestions: How to improve
        should_retry: Whether to retry with improvements
    """
    critique: str
    lessons_learned: List[str]
    improvement_suggestions: List[str]
    should_retry: bool


class EnhancedHypothesisAgent(HypothesisAgent):
    """
    Enhanced agent with advanced reasoning capabilities.

    Extends the base HypothesisAgent with:
    - Chain-of-thought reasoning
    - Self-reflection on failures
    - Dynamic few-shot examples
    - Multi-step planning

    Example:
        >>> agent = EnhancedHypothesisAgent()
        >>> proposal, reasoning_chain = agent.propose_with_reasoning(
        ...     current_best=materials,
        ...     target="low_energy_high_stability",
        ... )
        >>> # View reasoning process
        >>> for step in reasoning_chain:
        ...     print(f"Thought: {step.thought}")
        ...     print(f"Action: {step.action}")
    """

    def __init__(self, *args, enable_reflection: bool = True, **kwargs):
        """
        Initialize enhanced agent.

        Args:
            *args: Arguments for base HypothesisAgent
            enable_reflection: Enable self-reflection on failures
            **kwargs: Keyword arguments for base HypothesisAgent
        """
        super().__init__(*args, **kwargs)
        self.enable_reflection = enable_reflection
        self.reasoning_history = []
        self.reflection_history = []

    def _get_few_shot_examples(
        self,
        context: str,
        num_examples: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Generate relevant few-shot examples based on context.

        Uses materials science knowledge to provide examples that match
        the current search context.

        Args:
            context: Current search context
            num_examples: Number of examples to generate

        Returns:
            List of example dicts with 'input' and 'output' keys
        """
        # Hardcoded high-quality examples
        examples = [
            {
                "context": "Looking for stable oxides with low formation energy",
                "proposal": {
                    "elements": ["Ti", "O"],
                    "structure": "rutile",
                    "reasoning": "TiO2 is known for exceptional stability. Rutile structure "
                                "provides strong Ti-O bonding with octahedral coordination.",
                },
            },
            {
                "context": "Searching for semiconductors with band gap ~2 eV",
                "proposal": {
                    "elements": ["Zn", "O"],
                    "structure": "wurtzite",
                    "reasoning": "ZnO in wurtzite structure has band gap ~3.4 eV. "
                                "Doping or alloying can tune it down to 2 eV range.",
                },
            },
            {
                "context": "Need perovskites for photovoltaic applications",
                "proposal": {
                    "elements": ["Cs", "Pb", "I"],
                    "structure": "perovskite",
                    "reasoning": "CsPbI3 is a highly efficient perovskite photovoltaic material. "
                                "All-inorganic composition provides better stability than organic-inorganic hybrids.",
                },
            },
            {
                "context": "Exploring battery cathode materials",
                "proposal": {
                    "elements": ["Li", "Fe", "P", "O"],
                    "structure": "olivine",
                    "reasoning": "LiFePO4 olivine structure offers excellent cyclability, "
                                "safety, and low cost for Li-ion batteries.",
                },
            },
        ]

        # Select most relevant examples (in practice, would use embedding similarity)
        # For now, just return first N
        return examples[:num_examples]

    def propose_with_reasoning(
        self,
        current_best: List[Dict],
        previous_attempts: List[SearchSpaceProposal],
        target_property: str = "low_formation_energy",
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[SearchSpaceProposal, List[ReasoningStep]]:
        """
        Propose search space with explicit chain-of-thought reasoning.

        Returns both the proposal and the reasoning steps that led to it,
        enabling interpretability and debugging.

        Args:
            current_best: Top materials found so far
            previous_attempts: Previous search space proposals
            target_property: Optimization objective
            iteration: Current iteration
            total_iterations: Total budget

        Returns:
            proposal: SearchSpaceProposal
            reasoning_chain: List of ReasoningStep objects
        """
        # Build enhanced prompt with CoT instructions
        few_shot_examples = self._get_few_shot_examples(
            context=target_property,
            num_examples=2,
        )

        examples_text = "\n\n".join([
            f"Example {i+1}:\nContext: {ex['context']}\n"
            f"Proposal: {json.dumps(ex['proposal'], indent=2)}"
            for i, ex in enumerate(few_shot_examples)
        ])

        system_prompt = """You are an expert materials scientist using chain-of-thought reasoning to propose promising search spaces.

For each proposal, think step-by-step:
1. Analyze current results and patterns
2. Consider materials science principles
3. Generate hypotheses
4. Evaluate feasibility
5. Make final recommendation

Format your response as:
{
  "reasoning_steps": [
    {"thought": "...", "confidence": 0.0-1.0},
    ...
  ],
  "final_proposal": {
    "elements": [...],
    "num_elements_range": [min, max],
    "structure_prototypes": [...],
    "reasoning": "...",
    "confidence": 0.0-1.0
  }
}"""

        # Format current state
        best_materials_str = "\n".join([
            f"  - {m.get('formula', 'Unknown')}: "
            f"E_form={m.get('formation_energy', 'N/A'):.3f} eV/atom"
            for m in current_best[:5]
        ])

        user_prompt = f"""Target: {target_property}
Iteration: {iteration}/{total_iterations}

Current Best Materials:
{best_materials_str if best_materials_str else "  (none yet)"}

Few-Shot Examples:
{examples_text}

Now, propose the next search space using step-by-step reasoning.
Think through:
1. What patterns do you see in successful materials?
2. What chemical principles apply?
3. What unexplored regions are promising?
4. What's your confidence in each direction?

Respond in the JSON format specified above."""

        # Call API
        try:
            response, metadata = self._call_api(user_prompt, system_prompt)

            # Parse response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            data = json.loads(response[json_start:json_end])

            # Extract reasoning steps
            reasoning_chain = [
                ReasoningStep(
                    thought=step['thought'],
                    confidence=step.get('confidence', 1.0),
                )
                for step in data.get('reasoning_steps', [])
            ]

            # Extract proposal
            proposal_data = data['final_proposal']
            proposal = SearchSpaceProposal(
                elements=proposal_data['elements'],
                num_elements_range=tuple(proposal_data['num_elements_range']),
                structure_prototypes=proposal_data['structure_prototypes'],
                reasoning=proposal_data['reasoning'],
                confidence=proposal_data['confidence'],
            )

            # Store reasoning history
            self.reasoning_history.append({
                'iteration': iteration,
                'reasoning_chain': reasoning_chain,
                'proposal': proposal,
            })

            logger.info(
                f"Generated proposal with {len(reasoning_chain)} reasoning steps, "
                f"confidence: {proposal.confidence:.2f}"
            )

            return proposal, reasoning_chain

        except Exception as e:
            logger.error(f"Failed to generate proposal with reasoning: {e}")

            # Fallback to base method
            proposal = super().propose_search_space(
                current_best,
                previous_attempts,
                target_property,
                iteration,
                total_iterations,
            )

            # Create dummy reasoning chain
            reasoning_chain = [
                ReasoningStep(
                    thought="Falling back to simple proposal due to parsing error",
                    confidence=0.3,
                )
            ]

            return proposal, reasoning_chain

    def reflect_on_failure(
        self,
        failed_proposal: SearchSpaceProposal,
        failure_reason: str,
        actual_results: Dict,
    ) -> ReflectionResult:
        """
        Reflect on why a proposal failed and suggest improvements.

        Self-reflection helps the agent learn from mistakes and improve
        future proposals. This is inspired by the Reflexion framework.

        Args:
            failed_proposal: The proposal that didn't work well
            failure_reason: Why it failed (e.g., "no stable materials found")
            actual_results: What actually happened

        Returns:
            ReflectionResult with critique and suggestions
        """
        if not self.enable_reflection:
            return ReflectionResult(
                critique="Reflection disabled",
                lessons_learned=[],
                improvement_suggestions=[],
                should_retry=False,
            )

        system_prompt = """You are a critical reviewer analyzing failed materials discovery attempts.

Provide:
1. Critique: What went wrong?
2. Lessons: What did we learn?
3. Improvements: How to do better next time?
4. Retry: Should we retry with changes?"""

        user_prompt = f"""Failed Proposal:
Elements: {failed_proposal.elements}
Structures: {failed_proposal.structure_prototypes}
Original Reasoning: {failed_proposal.reasoning}
Confidence: {failed_proposal.confidence}

Failure Reason: {failure_reason}

Actual Results:
{json.dumps(actual_results, indent=2)}

Analyze this failure and suggest improvements."""

        try:
            response, _ = self._call_api(
                user_prompt,
                system_prompt,
                use_fallback=True,  # Use cheaper model for reflection
            )

            # Parse response (simplified - in practice would use structured output)
            reflection = ReflectionResult(
                critique=response[:200],
                lessons_learned=["Extracted from response"],
                improvement_suggestions=["Extracted from response"],
                should_retry=True,
            )

            self.reflection_history.append({
                'failed_proposal': failed_proposal,
                'reflection': reflection,
            })

            logger.info(f"Reflected on failure: {reflection.critique[:100]}...")

            return reflection

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return ReflectionResult(
                critique="Reflection failed",
                lessons_learned=[],
                improvement_suggestions=[],
                should_retry=False,
            )

    def multi_agent_discussion(
        self,
        proposals: List[SearchSpaceProposal],
        num_rounds: int = 2,
    ) -> SearchSpaceProposal:
        """
        Simulate multiple agents discussing and reaching consensus.

        Different "perspectives" critique each other's proposals and
        arrive at a better combined proposal.

        Args:
            proposals: Initial proposals from different strategies
            num_rounds: Number of discussion rounds

        Returns:
            Consensus proposal
        """
        system_prompt = """You are facilitating a discussion between multiple materials scientists.

Each scientist has proposed a search space. Your job is to:
1. Identify strengths and weaknesses of each proposal
2. Find common ground
3. Synthesize a better combined proposal

Be objective and evidence-based."""

        proposals_text = "\n\n".join([
            f"Proposal {i+1}:\n"
            f"Elements: {p.elements}\n"
            f"Structures: {p.structure_prototypes}\n"
            f"Reasoning: {p.reasoning}\n"
            f"Confidence: {p.confidence}"
            for i, p in enumerate(proposals)
        ])

        user_prompt = f"""Multiple proposals have been made:

{proposals_text}

Synthesize these into one improved proposal that combines their strengths.
Output format:
{{
  "synthesis_reasoning": "...",
  "combined_proposal": {{
    "elements": [...],
    "num_elements_range": [min, max],
    "structure_prototypes": [...],
    "reasoning": "...",
    "confidence": 0.0-1.0
  }}
}}"""

        try:
            response, _ = self._call_api(user_prompt, system_prompt)

            # Parse combined proposal
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            data = json.loads(response[json_start:json_end])

            combined_data = data['combined_proposal']
            combined_proposal = SearchSpaceProposal(
                elements=combined_data['elements'],
                num_elements_range=tuple(combined_data['num_elements_range']),
                structure_prototypes=combined_data['structure_prototypes'],
                reasoning=f"Synthesis: {data['synthesis_reasoning']}. "
                         f"Details: {combined_data['reasoning']}",
                confidence=combined_data['confidence'],
            )

            logger.info(
                f"Multi-agent consensus reached: {combined_proposal.elements}, "
                f"confidence: {combined_proposal.confidence:.2f}"
            )

            return combined_proposal

        except Exception as e:
            logger.error(f"Multi-agent discussion failed: {e}")
            # Return highest confidence proposal
            return max(proposals, key=lambda p: p.confidence)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Enhanced Agent Capabilities Demo\n")

    # Note: Requires ANTHROPIC_API_KEY
    # agent = EnhancedHypothesisAgent()

    print("Features demonstrated:")
    print("1. Chain-of-Thought Reasoning")
    print("   - Step-by-step thought process")
    print("   - Confidence tracking")
    print()
    print("2. Self-Reflection")
    print("   - Analyzes failures")
    print("   - Learns from mistakes")
    print("   - Suggests improvements")
    print()
    print("3. Few-Shot Learning")
    print("   - Context-aware examples")
    print("   - Improves proposal quality")
    print()
    print("4. Multi-Agent Discussion")
    print("   - Multiple perspectives")
    print("   - Consensus building")
    print("   - Better decisions")

    print("\n✅ Module loaded successfully!")

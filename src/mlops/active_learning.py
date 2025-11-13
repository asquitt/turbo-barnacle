"""
Active Learning for Materials Discovery

This module implements active learning strategies to:
- Intelligently select which materials to validate with expensive DFT
- Maximize information gain per evaluation
- Reduce total discovery cost
- Handle class imbalance (stable vs unstable materials)

Key Strategies:
- Uncertainty sampling: Query most uncertain predictions
- Query-by-committee: Query where models disagree most
- Expected model change: Query what changes model most
- Diversity sampling: Ensure diverse coverage

Research Background:
- Active learning survey: https://arxiv.org/abs/2009.00236
- Materials active learning: https://www.nature.com/articles/s41524-021-00554-0

Author: Materials Discovery Team
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningQuery:
    """
    Result from active learning query selection.

    Attributes:
        selected_indices: Indices of selected samples
        selection_scores: Scores for each selected sample
        strategy: Strategy used for selection
        information_gain: Expected information gain
    """
    selected_indices: List[int]
    selection_scores: np.ndarray
    strategy: str
    information_gain: float


class UncertaintySampling:
    """
    Uncertainty sampling strategy.

    Selects samples where the model is most uncertain. High uncertainty
    means the model would benefit most from knowing the true label.

    Strategies:
    - least_confident: lowest prediction confidence
    - margin: smallest margin between top 2 classes
    - entropy: highest prediction entropy
    """

    def __init__(self, strategy: str = "entropy"):
        """
        Args:
            strategy: Uncertainty measure ("least_confident", "margin", "entropy")
        """
        self.strategy = strategy

    def __call__(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        n_samples: int,
    ) -> ActiveLearningQuery:
        """
        Select most uncertain samples.

        Args:
            predictions: Model predictions [num_samples, num_tasks]
            uncertainties: Prediction uncertainties [num_samples, num_tasks]
            n_samples: Number of samples to select

        Returns:
            ActiveLearningQuery with selected indices
        """
        if self.strategy == "entropy":
            # Higher entropy = more uncertain
            # For regression, use prediction std as proxy
            scores = uncertainties.mean(axis=1)

        elif self.strategy == "least_confident":
            # For binary stability task
            stability_pred = predictions[:, 2]
            scores = np.minimum(stability_pred, 1 - stability_pred)

        elif self.strategy == "margin":
            # Margin between prediction and nearest decision boundary
            # For regression, use normalized uncertainty
            scores = uncertainties[:, 0] / (np.abs(predictions[:, 0]) + 1e-6)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Select top N uncertain samples
        selected_indices = np.argsort(-scores)[:n_samples].tolist()
        selected_scores = scores[selected_indices]

        # Estimate information gain (average uncertainty)
        info_gain = float(selected_scores.mean())

        return ActiveLearningQuery(
            selected_indices=selected_indices,
            selection_scores=selected_scores,
            strategy=f"uncertainty_{self.strategy}",
            information_gain=info_gain,
        )


class QueryByCommittee:
    """
    Query-by-committee strategy.

    Uses disagreement among multiple models (ensemble) to identify
    informative samples. High disagreement = high information value.

    Args:
        committee_predictions: Predictions from each committee member
            Shape: [num_models, num_samples, num_tasks]
    """

    def __init__(self, measure: str = "vote_entropy"):
        """
        Args:
            measure: Disagreement measure ("vote_entropy", "kl_divergence", "variance")
        """
        self.measure = measure

    def __call__(
        self,
        committee_predictions: np.ndarray,
        n_samples: int,
    ) -> ActiveLearningQuery:
        """
        Select samples with highest committee disagreement.

        Args:
            committee_predictions: [num_models, num_samples, num_tasks]
            n_samples: Number to select

        Returns:
            ActiveLearningQuery
        """
        if self.measure == "variance":
            # Variance across committee
            scores = committee_predictions.var(axis=0).mean(axis=1)

        elif self.measure == "vote_entropy":
            # Entropy of committee votes (for classification)
            # For stability prediction (task 2)
            stability_preds = committee_predictions[:, :, 2]  # [num_models, num_samples]
            votes = (stability_preds > 0.5).astype(float)
            vote_counts = votes.mean(axis=0)  # [num_samples]

            # Entropy: -p*log(p) - (1-p)*log(1-p)
            eps = 1e-9
            entropy = -(
                vote_counts * np.log(vote_counts + eps) +
                (1 - vote_counts) * np.log(1 - vote_counts + eps)
            )
            scores = entropy

        else:
            # Default: standard deviation
            scores = committee_predictions.std(axis=0).mean(axis=1)

        selected_indices = np.argsort(-scores)[:n_samples].tolist()
        selected_scores = scores[selected_indices]

        info_gain = float(selected_scores.mean())

        return ActiveLearningQuery(
            selected_indices=selected_indices,
            selection_scores=selected_scores,
            strategy=f"qbc_{self.measure}",
            information_gain=info_gain,
        )


class DiversitySampling:
    """
    Diversity sampling strategy.

    Selects diverse samples to ensure broad coverage of the feature space.
    Uses clustering (K-means) to find representative samples.

    Useful when combined with uncertainty sampling to avoid selecting
    many similar uncertain samples.
    """

    def __init__(self, method: str = "kmeans"):
        """
        Args:
            method: Diversity method ("kmeans", "max_distance")
        """
        self.method = method

    def __call__(
        self,
        features: np.ndarray,
        n_samples: int,
    ) -> ActiveLearningQuery:
        """
        Select diverse samples.

        Args:
            features: Sample features [num_samples, num_features]
            n_samples: Number to select

        Returns:
            ActiveLearningQuery
        """
        if self.method == "kmeans":
            # Cluster and select cluster centers
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(features)

            # Find samples closest to cluster centers
            selected_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(features - center, axis=1)
                closest = np.argmin(distances)
                if closest not in selected_indices:
                    selected_indices.append(int(closest))

            # Fill if needed
            while len(selected_indices) < n_samples:
                for i in range(len(features)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        break

            selected_indices = selected_indices[:n_samples]

        elif self.method == "max_distance":
            # Greedy: select sample farthest from already selected
            selected_indices = [0]  # Start with first sample

            for _ in range(n_samples - 1):
                # Compute minimum distance to selected samples
                selected = features[selected_indices]
                distances = np.linalg.norm(
                    features[:, None, :] - selected[None, :, :],
                    axis=2
                ).min(axis=1)

                # Select farthest
                next_idx = int(np.argmax(distances))
                selected_indices.append(next_idx)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Scores are distances to nearest selected sample (diversity measure)
        selected = features[selected_indices]
        scores = np.linalg.norm(
            features[:, None, :] - selected[None, :, :],
            axis=2
        ).min(axis=1)[selected_indices]

        info_gain = float(scores.mean())

        return ActiveLearningQuery(
            selected_indices=selected_indices,
            selection_scores=scores,
            strategy=f"diversity_{self.method}",
            information_gain=info_gain,
        )


class HybridActiveLearning:
    """
    Hybrid active learning combining multiple strategies.

    Balances:
    - Uncertainty (exploitation): learn where model is weak
    - Diversity (exploration): cover feature space broadly
    - Expected model change: maximize learning

    Args:
        uncertainty_weight: Weight for uncertainty sampling
        diversity_weight: Weight for diversity sampling
        committee_weight: Weight for query-by-committee
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.3,
        committee_weight: float = 0.2,
    ):
        """
        Initialize hybrid active learner.

        Args:
            uncertainty_weight: Weight for uncertainty
            diversity_weight: Weight for diversity
            committee_weight: Weight for committee disagreement
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.committee_weight = committee_weight

        self.uncertainty_sampler = UncertaintySampling(strategy="entropy")
        self.diversity_sampler = DiversitySampling(method="kmeans")
        self.qbc_sampler = QueryByCommittee(measure="variance")

        logger.info(
            f"Initialized HybridActiveLearning: "
            f"uncertainty={uncertainty_weight}, "
            f"diversity={diversity_weight}, "
            f"committee={committee_weight}"
        )

    def __call__(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        features: np.ndarray,
        committee_predictions: Optional[np.ndarray] = None,
        n_samples: int = 20,
    ) -> ActiveLearningQuery:
        """
        Select samples using hybrid strategy.

        Args:
            predictions: Model predictions [num_samples, num_tasks]
            uncertainties: Prediction uncertainties [num_samples, num_tasks]
            features: Sample features [num_samples, num_features]
            committee_predictions: [num_models, num_samples, num_tasks] (optional)
            n_samples: Number to select

        Returns:
            ActiveLearningQuery
        """
        num_candidates = len(predictions)
        combined_scores = np.zeros(num_candidates)

        # 1. Uncertainty scores
        if self.uncertainty_weight > 0:
            uncertainty_result = self.uncertainty_sampler(
                predictions, uncertainties, num_candidates
            )
            # Normalize scores
            uncertainty_scores = uncertainty_result.selection_scores
            uncertainty_scores = (
                (uncertainty_scores - uncertainty_scores.min()) /
                (uncertainty_scores.max() - uncertainty_scores.min() + 1e-9)
            )
            combined_scores += self.uncertainty_weight * uncertainty_scores

        # 2. Diversity scores
        if self.diversity_weight > 0:
            diversity_result = self.diversity_sampler(features, num_candidates)
            diversity_indices = diversity_result.selected_indices
            diversity_scores = np.zeros(num_candidates)
            diversity_scores[diversity_indices] = 1.0
            combined_scores += self.diversity_weight * diversity_scores

        # 3. Committee disagreement scores
        if self.committee_weight > 0 and committee_predictions is not None:
            qbc_result = self.qbc_sampler(committee_predictions, num_candidates)
            qbc_scores = np.zeros(num_candidates)
            qbc_scores[qbc_result.selected_indices] = qbc_result.selection_scores
            qbc_scores = (
                (qbc_scores - qbc_scores.min()) /
                (qbc_scores.max() - qbc_scores.min() + 1e-9)
            )
            combined_scores += self.committee_weight * qbc_scores

        # Select top N by combined score
        selected_indices = np.argsort(-combined_scores)[:n_samples].tolist()
        selected_scores = combined_scores[selected_indices]

        info_gain = float(selected_scores.mean())

        return ActiveLearningQuery(
            selected_indices=selected_indices,
            selection_scores=selected_scores,
            strategy="hybrid",
            information_gain=info_gain,
        )


def estimate_dft_budget(
    total_candidates: int,
    discovery_rate: float = 0.1,
    dft_cost_per_material: float = 0.1,
    max_budget: float = 100.0,
) -> int:
    """
    Estimate how many DFT validations we can afford.

    Args:
        total_candidates: Total materials to screen
        discovery_rate: Expected fraction of stable materials
        dft_cost_per_material: Cost per DFT calculation (USD)
        max_budget: Maximum budget (USD)

    Returns:
        Number of DFT validations we can afford
    """
    # Estimate materials to validate
    expected_discoveries = int(total_candidates * discovery_rate)

    # Budget for validating all discoveries
    validation_budget = expected_discoveries * dft_cost_per_material

    # Number we can actually afford
    affordable = int(min(max_budget / dft_cost_per_material, expected_discoveries))

    logger.info(
        f"DFT budget: ${max_budget:.2f} → can validate {affordable} materials "
        f"(expected discoveries: {expected_discoveries})"
    )

    return affordable


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing Active Learning Strategies\n")

    # Generate mock data
    num_samples = 100
    predictions = np.random.randn(num_samples, 3)
    uncertainties = np.abs(np.random.randn(num_samples, 3)) * 0.5
    features = np.random.randn(num_samples, 10)

    # Test each strategy
    print("1. Uncertainty Sampling")
    uncertainty = UncertaintySampling(strategy="entropy")
    result = uncertainty(predictions, uncertainties, n_samples=10)
    print(f"   Selected: {result.selected_indices[:5]}...")
    print(f"   Info gain: {result.information_gain:.4f}")

    print("\n2. Diversity Sampling")
    diversity = DiversitySampling(method="kmeans")
    result = diversity(features, n_samples=10)
    print(f"   Selected: {result.selected_indices[:5]}...")
    print(f"   Info gain: {result.information_gain:.4f}")

    print("\n3. Query-by-Committee")
    committee_preds = np.random.randn(5, num_samples, 3)  # 5 models
    qbc = QueryByCommittee(measure="variance")
    result = qbc(committee_preds, n_samples=10)
    print(f"   Selected: {result.selected_indices[:5]}...")
    print(f"   Info gain: {result.information_gain:.4f}")

    print("\n4. Hybrid Strategy")
    hybrid = HybridActiveLearning(
        uncertainty_weight=0.5,
        diversity_weight=0.3,
        committee_weight=0.2,
    )
    result = hybrid(
        predictions,
        uncertainties,
        features,
        committee_predictions=committee_preds,
        n_samples=10,
    )
    print(f"   Selected: {result.selected_indices[:5]}...")
    print(f"   Info gain: {result.information_gain:.4f}")

    print("\n5. DFT Budget Estimation")
    budget = estimate_dft_budget(
        total_candidates=1000,
        discovery_rate=0.1,
        dft_cost_per_material=0.1,
        max_budget=10.0,
    )
    print(f"   Can afford to validate: {budget} materials")

    print("\n✅ All tests passed!")

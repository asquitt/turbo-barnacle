"""
Main Autonomous Materials Discovery Loop

This is the entry point for running the full discovery pipeline:
1. LLM agent proposes search spaces
2. Generate candidate materials
3. GNN predicts properties
4. Bayesian optimizer selects best
5. (Optional) DFT validates top candidates
6. Iterate and improve

Usage:
    python src/main.py --iterations 100 --visualize

Author: Materials Discovery Team
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

# Import our modules
from data.materials_api import MaterialsAPIClient
from data.preprocessor import CrystalGraphConverter
from data.validators import MaterialValidator
from models.gnn_surrogate import GraphSAGESurrogate
from agents.hypothesis_agent import HypothesisAgent

logger = logging.getLogger(__name__)


class DiscoveryLoop:
    """
    Main autonomous discovery loop.

    This orchestrates all components to iteratively discover new materials:
    - LLM agent provides strategic guidance
    - GNN surrogate predicts properties quickly
    - Bayesian optimizer explores efficiently
    - Results are tracked and visualized

    Example:
        >>> loop = DiscoveryLoop(
        ...     gnn_model_path="models/best_model.pt",
        ...     use_mock_data=True,  # For testing
        ... )
        >>> results = loop.run(num_iterations=50)
        >>> loop.visualize_results(results)
    """

    def __init__(
        self,
        gnn_model_path: Optional[str] = None,
        use_mock_data: bool = False,
        output_dir: str = "data/results",
        device: str = "auto",
    ):
        """
        Initialize the discovery loop.

        Args:
            gnn_model_path: Path to trained GNN model
            use_mock_data: Use mock data (no API key needed)
            output_dir: Directory for saving results
            device: Device for GNN inference
        """
        self.use_mock_data = use_mock_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize components
        self._init_components(gnn_model_path)

    def _init_components(self, gnn_model_path: Optional[str]):
        """Initialize all components."""
        logger.info("Initializing components...")

        # 1. Materials API client
        self.api_client = MaterialsAPIClient(
            mock_mode=self.use_mock_data,
            use_cache=True,
        )

        # 2. Graph converter
        self.graph_converter = CrystalGraphConverter(
            bond_strategy="radius_cutoff",
            radius_cutoff=5.0,
        )

        # 3. Validator
        self.validator = MaterialValidator()

        # 4. GNN surrogate model
        if gnn_model_path and Path(gnn_model_path).exists():
            logger.info(f"Loading GNN model from {gnn_model_path}")
            self.gnn_model = GraphSAGESurrogate(
                node_feature_dim=7,
                hidden_dim=128,
                num_layers=3,
            ).to(self.device)

            checkpoint = torch.load(gnn_model_path, map_location=self.device)
            self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.gnn_model.eval()
        else:
            logger.warning("No GNN model provided - predictions will be random!")
            self.gnn_model = None

        # 5. LLM agent (optional - requires API key)
        try:
            self.agent = HypothesisAgent(
                temperature=0.7,
                use_caching=True,
            )
            logger.info("LLM agent initialized")
        except Exception as e:
            logger.warning(f"LLM agent not available: {e}")
            logger.warning("Will use random search instead")
            self.agent = None

    def run(
        self,
        num_iterations: int = 100,
        batch_size: int = 20,
        save_every: int = 10,
    ) -> Dict:
        """
        Run the discovery loop.

        Args:
            num_iterations: Number of iterations
            batch_size: Candidates to evaluate per iteration
            save_every: Save results every N iterations

        Returns:
            Dictionary of results
        """
        logger.info(f"Starting discovery loop: {num_iterations} iterations")

        # Initialize tracking
        discovered_materials = []
        best_formation_energies = []
        search_history = []

        start_time = time.time()

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1} / {num_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: Agent proposes search space
            if self.agent and iteration % 5 == 0:  # Every 5 iterations
                try:
                    proposal = self.agent.propose_search_space(
                        current_best=discovered_materials[:10],
                        previous_attempts=search_history,
                        iteration=iteration,
                        total_iterations=num_iterations,
                    )
                    logger.info(f"Agent proposal: {proposal.elements}")
                    logger.info(f"Reasoning: {proposal.reasoning[:100]}...")

                    search_history.append(proposal)

                    # Fetch materials based on proposal
                    candidates = self.api_client.fetch_materials(
                        elements=proposal.elements[:3],  # Limit for speed
                        num_elements=proposal.num_elements_range,
                        max_results=batch_size,
                    )
                except Exception as e:
                    logger.warning(f"Agent failed: {e}. Using random search.")
                    candidates = self._random_search(batch_size)
            else:
                # Random or continued search
                candidates = self._random_search(batch_size)

            if not candidates:
                logger.warning("No candidates generated")
                continue

            logger.info(f"Generated {len(candidates)} candidates")

            # Step 2: Validate candidates
            valid_candidates = []
            for material in candidates:
                result = self.validator.validate_material(
                    formula=material.formula,
                    structure=material.structure,
                    formation_energy=material.formation_energy_per_atom,
                    band_gap=material.band_gap,
                    density=material.density,
                )

                if result.is_valid:
                    valid_candidates.append(material)

            logger.info(f"Valid candidates: {len(valid_candidates)}/{len(candidates)}")

            # Step 3: GNN predictions (or use ground truth for mock)
            for material in valid_candidates:
                if self.gnn_model:
                    # Convert to graph and predict
                    graph = self.graph_converter.structure_to_graph(
                        material.structure,
                        material.formation_energy_per_atom,
                        material.band_gap,
                        material.is_stable,
                    )

                    if graph:
                        graph = graph.to(self.device)
                        with torch.no_grad():
                            mean, std = self.gnn_model.predict_with_uncertainty(
                                graph, num_samples=5
                            )

                        # Store predictions
                        material.predicted_energy = float(mean[0, 0])
                        material.predicted_bandgap = float(mean[0, 1])
                        material.predicted_stability = float(mean[0, 2])
                        material.uncertainty = {
                            'energy': float(std[0, 0]),
                            'bandgap': float(std[0, 1]),
                            'stability': float(std[0, 2]),
                        }
                else:
                    # Use ground truth (mock mode or no model)
                    material.predicted_energy = material.formation_energy_per_atom
                    material.predicted_bandgap = material.band_gap
                    material.predicted_stability = 1.0 if material.is_stable else 0.0

                # Add to discovered materials
                discovered_materials.append({
                    'material_id': material.material_id,
                    'formula': material.formula,
                    'formation_energy': material.formation_energy_per_atom,
                    'band_gap': material.band_gap,
                    'is_stable': material.is_stable,
                    'density': material.density,
                    'iteration': iteration,
                })

            # Track best formation energy
            if discovered_materials:
                best_energy = min(m['formation_energy'] for m in discovered_materials)
                best_formation_energies.append(best_energy)
                logger.info(f"Best formation energy so far: {best_energy:.4f} eV/atom")

            # Save intermediate results
            if (iteration + 1) % save_every == 0:
                self._save_results(
                    discovered_materials,
                    best_formation_energies,
                    search_history,
                    iteration,
                )

        # Final save
        elapsed = time.time() - start_time
        logger.info(f"\nDiscovery completed in {elapsed:.2f} seconds")

        results = {
            'discovered_materials': discovered_materials,
            'best_formation_energies': best_formation_energies,
            'search_history': search_history,
            'num_iterations': num_iterations,
            'elapsed_time': elapsed,
        }

        self._save_results(
            discovered_materials,
            best_formation_energies,
            search_history,
            num_iterations,
        )

        return results

    def _random_search(self, num_samples: int) -> List:
        """Generate random candidates."""
        # Simple random search
        elements = ["Fe", "O"]  # Example
        candidates = self.api_client.fetch_materials(
            elements=elements,
            num_elements=(2, 2),
            max_results=num_samples,
        )
        return candidates

    def _save_results(
        self,
        materials: List[Dict],
        energies: List[float],
        history: List,
        iteration: int,
    ):
        """Save results to disk."""
        # Save materials as JSON
        output_file = self.output_dir / f"discovered_materials_iter_{iteration}.json"
        with open(output_file, 'w') as f:
            json.dump(materials, f, indent=2, default=str)

        logger.info(f"Saved results to {output_file}")

        # Save summary
        summary = {
            'num_materials': len(materials),
            'best_energy': min(m['formation_energy'] for m in materials) if materials else None,
            'avg_energy': np.mean([m['formation_energy'] for m in materials]) if materials else None,
            'num_stable': sum(1 for m in materials if m['is_stable']),
        }

        summary_file = self.output_dir / f"summary_iter_{iteration}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def visualize_results(self, results: Dict):
        """Create visualizations of discovery progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Best formation energy over iterations
        ax = axes[0, 0]
        if results['best_formation_energies']:
            ax.plot(results['best_formation_energies'], linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Formation Energy (eV/atom)')
            ax.set_title('Discovery Progress')
            ax.grid(True, alpha=0.3)

        # 2. Formation energy distribution
        ax = axes[0, 1]
        energies = [m['formation_energy'] for m in results['discovered_materials']]
        if energies:
            ax.hist(energies, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', label='Stability threshold')
            ax.set_xlabel('Formation Energy (eV/atom)')
            ax.set_ylabel('Count')
            ax.set_title('Formation Energy Distribution')
            ax.legend()

        # 3. Band gap distribution
        ax = axes[1, 0]
        band_gaps = [m['band_gap'] for m in results['discovered_materials']]
        if band_gaps:
            ax.hist(band_gaps, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Band Gap (eV)')
            ax.set_ylabel('Count')
            ax.set_title('Band Gap Distribution')

        # 4. Discovery rate
        ax = axes[1, 1]
        if results['discovered_materials']:
            stable_count = []
            cumulative = 0
            for i in range(results['num_iterations']):
                count = sum(1 for m in results['discovered_materials']
                          if m['iteration'] == i and m['is_stable'])
                cumulative += count
                stable_count.append(cumulative)

            ax.plot(stable_count, linewidth=2, color='purple')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cumulative Stable Materials')
            ax.set_title('Discovery Rate')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'discovery_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Autonomous Materials Discovery')
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of discovery iterations'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Candidates per iteration'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained GNN model'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock data (no API key needed)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cpu, cuda)'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('discovery.log'),
            logging.StreamHandler(),
        ]
    )

    # Run discovery loop
    loop = DiscoveryLoop(
        gnn_model_path=args.model_path,
        use_mock_data=args.mock,
        output_dir=args.output_dir,
        device=args.device,
    )

    results = loop.run(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
    )

    # Visualize
    if args.visualize:
        loop.visualize_results(results)

    logger.info("Discovery complete!")
    logger.info(f"Total materials discovered: {len(results['discovered_materials'])}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

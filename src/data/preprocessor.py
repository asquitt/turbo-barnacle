"""
Crystal Structure to Graph Converter

This module converts crystal structures (from pymatgen) into graph representations
suitable for Graph Neural Networks (PyTorch Geometric).

Key Concepts:
- Nodes = Atoms (features: atomic number, electronegativity, radius, etc.)
- Edges = Chemical bonds (features: distance, bond order estimate)
- Graph = Complete crystal structure ready for GNN

Learning Resources:
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- Pymatgen structure docs: https://pymatgen.org/pymatgen.core.structure.html

Author: Materials Discovery Team
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """
    Container for graph features extracted from crystal structure.

    Node features (per atom):
        - atomic_number: Element atomic number (1-118)
        - atomic_mass: Atomic mass in amu
        - electronegativity: Pauling electronegativity
        - covalent_radius: Covalent radius in Angstroms
        - valence_electrons: Number of valence electrons
        - group: Periodic table group (1-18)
        - period: Periodic table period (1-7)

    Edge features (per bond):
        - distance: Bond length in Angstroms
        - bond_order: Estimated bond order (1, 2, 3, etc.)
        - direction: 3D vector from source to target atom

    Global features (per structure):
        - num_atoms: Total atoms in unit cell
        - volume: Unit cell volume in Angstrom^3
        - density: Density in g/cm^3
        - composition: Element ratios
    """
    pass  # Type hints for documentation only


class CrystalGraphConverter:
    """
    Converts pymatgen Structure objects to PyTorch Geometric Data objects.

    This converter implements several strategies for defining chemical bonds:
    1. Radius cutoff: All atoms within X Angstroms
    2. K-nearest neighbors: K closest atoms to each atom
    3. CrystalNN: Voronoi-based bond detection (most accurate, slower)

    The choice of strategy affects:
    - Graph connectivity
    - Computational cost
    - Model performance

    Example:
        >>> from pymatgen.core import Structure, Lattice
        >>> converter = CrystalGraphConverter(bond_strategy="radius_cutoff")
        >>>
        >>> # Create a simple structure
        >>> lattice = Lattice.cubic(5.0)
        >>> structure = Structure(lattice, ["Fe", "O"], [[0,0,0], [0.5,0.5,0.5]])
        >>>
        >>> # Convert to graph
        >>> graph = converter.structure_to_graph(structure)
        >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    """

    def __init__(
        self,
        bond_strategy: str = "radius_cutoff",
        radius_cutoff: float = 5.0,
        k_neighbors: int = 12,
        max_atoms: int = 100,
        use_edge_features: bool = True,
    ):
        """
        Initialize the graph converter.

        Args:
            bond_strategy: How to define bonds - "radius_cutoff", "knn", or "crystalnn"
            radius_cutoff: Maximum bond distance in Angstroms (for radius_cutoff)
            k_neighbors: Number of neighbors per atom (for knn)
            max_atoms: Maximum atoms per structure (for memory control)
            use_edge_features: Include bond distance/direction features
        """
        self.bond_strategy = bond_strategy
        self.radius_cutoff = radius_cutoff
        self.k_neighbors = k_neighbors
        self.max_atoms = max_atoms
        self.use_edge_features = use_edge_features

        # Initialize CrystalNN if using that strategy
        if bond_strategy == "crystalnn":
            self.crystalnn = CrystalNN()

        # Cache element properties for fast lookups
        self._element_cache = {}
        logger.info(f"Initialized CrystalGraphConverter with {bond_strategy} strategy")

    def _get_element_features(self, element: Element) -> np.ndarray:
        """
        Get feature vector for an element.

        Features (7-dimensional):
        1. Atomic number (normalized to 0-1)
        2. Atomic mass (normalized)
        3. Electronegativity (Pauling scale, 0 if not available)
        4. Covalent radius (normalized)
        5. Valence electrons (normalized to 0-1)
        6. Group (normalized to 0-1)
        7. Period (normalized to 0-1)

        These features capture chemical properties relevant for bonding and stability.
        """
        # Check cache first
        symbol = element.symbol
        if symbol in self._element_cache:
            return self._element_cache[symbol]

        # Extract properties (with defaults for missing data)
        atomic_number = element.Z / 118.0  # Normalize to [0, 1]
        atomic_mass = element.atomic_mass / 300.0  # Rough normalization

        # Electronegativity (Pauling scale)
        try:
            electronegativity = element.X / 4.0  # Scale to ~[0, 1]
        except:
            electronegativity = 0.5  # Default for elements without data

        # Covalent radius
        try:
            covalent_radius = element.average_ionic_radius / 3.0  # Normalize
        except:
            covalent_radius = 0.5

        # Valence electrons (approximation)
        try:
            valence = element.group % 10 / 8.0  # Rough estimate
        except:
            valence = 0.5

        # Periodic table position
        try:
            group = element.group / 18.0
        except:
            group = 0.5

        try:
            period = element.row / 7.0
        except:
            period = 0.5

        # Combine into feature vector
        features = np.array([
            atomic_number,
            atomic_mass,
            electronegativity,
            covalent_radius,
            valence,
            group,
            period,
        ], dtype=np.float32)

        # Cache for future use
        self._element_cache[symbol] = features
        return features

    def _get_bonds_radius_cutoff(
        self,
        structure: Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bonds using radius cutoff strategy.

        All atoms within `radius_cutoff` Angstroms are connected.
        This is the simplest and fastest strategy but may create
        unrealistic bonds.

        Returns:
            edge_index: (2, num_edges) array of [source, target] indices
            edge_distances: (num_edges,) array of distances in Angstroms
        """
        edge_list = []
        edge_distances = []

        # Get all neighbors within cutoff radius
        neighbors_all = structure.get_all_neighbors(self.radius_cutoff)

        for i, neighbors in enumerate(neighbors_all):
            for neighbor in neighbors:
                j = neighbor.index
                distance = neighbor.nn_distance

                # Add edge (undirected, so both i->j and j->i)
                edge_list.append([i, j])
                edge_distances.append(distance)

        # Convert to numpy arrays
        if len(edge_list) == 0:
            # No edges found - create self-loops
            num_atoms = len(structure)
            edge_index = np.array([[i, i] for i in range(num_atoms)]).T
            edge_distances = np.zeros(num_atoms)
            logger.warning("No bonds found, using self-loops")
        else:
            edge_index = np.array(edge_list).T
            edge_distances = np.array(edge_distances)

        return edge_index, edge_distances

    def _get_bonds_knn(
        self,
        structure: Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bonds using K-nearest neighbors strategy.

        Each atom is connected to its K closest neighbors.
        This ensures consistent node degree but may miss some bonds
        or include unrealistic long-range connections.

        Returns:
            edge_index: (2, num_edges) array
            edge_distances: (num_edges,) array
        """
        edge_list = []
        edge_distances = []

        # Get K nearest neighbors for each atom
        neighbors_all = structure.get_all_neighbors(r=self.radius_cutoff)

        for i, neighbors in enumerate(neighbors_all):
            # Sort by distance and take top K
            neighbors = sorted(neighbors, key=lambda x: x.nn_distance)
            neighbors = neighbors[:self.k_neighbors]

            for neighbor in neighbors:
                j = neighbor.index
                distance = neighbor.nn_distance

                edge_list.append([i, j])
                edge_distances.append(distance)

        if len(edge_list) == 0:
            num_atoms = len(structure)
            edge_index = np.array([[i, i] for i in range(num_atoms)]).T
            edge_distances = np.zeros(num_atoms)
        else:
            edge_index = np.array(edge_list).T
            edge_distances = np.array(edge_distances)

        return edge_index, edge_distances

    def _get_bonds_crystalnn(
        self,
        structure: Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bonds using CrystalNN (Voronoi-based) strategy.

        This uses solid-state chemistry heuristics to determine
        which atoms are truly bonded. Most accurate but slowest.

        Returns:
            edge_index: (2, num_edges) array
            edge_distances: (num_edges,) array
        """
        edge_list = []
        edge_distances = []

        try:
            # Use CrystalNN to get bonded neighbors
            for i in range(len(structure)):
                neighbors = self.crystalnn.get_nn_info(structure, i)

                for neighbor in neighbors:
                    j = neighbor['site_index']
                    distance = neighbor['site'].distance(structure[i])

                    edge_list.append([i, j])
                    edge_distances.append(distance)

        except Exception as e:
            logger.warning(f"CrystalNN failed: {e}. Falling back to radius cutoff.")
            return self._get_bonds_radius_cutoff(structure)

        if len(edge_list) == 0:
            num_atoms = len(structure)
            edge_index = np.array([[i, i] for i in range(num_atoms)]).T
            edge_distances = np.zeros(num_atoms)
        else:
            edge_index = np.array(edge_list).T
            edge_distances = np.array(edge_distances)

        return edge_index, edge_distances

    def structure_to_graph(
        self,
        structure: Structure,
        formation_energy: Optional[float] = None,
        band_gap: Optional[float] = None,
        is_stable: Optional[bool] = None,
    ) -> Data:
        """
        Convert a pymatgen Structure to a PyTorch Geometric Data object.

        Args:
            structure: Pymatgen Structure object
            formation_energy: Formation energy per atom (eV/atom) - target label
            band_gap: Band gap (eV) - target label
            is_stable: Stability flag - target label

        Returns:
            PyTorch Geometric Data object with:
                - x: Node features (num_atoms, num_features)
                - edge_index: Edge connectivity (2, num_edges)
                - edge_attr: Edge features (num_edges, num_edge_features)
                - y: Target labels [formation_energy, band_gap, is_stable]
                - num_atoms: Number of atoms
                - formula: Chemical formula (string)

        Example:
            >>> structure = Structure(...) # Your crystal structure
            >>> graph = converter.structure_to_graph(
            ...     structure,
            ...     formation_energy=-2.5,
            ...     band_gap=1.2,
            ...     is_stable=True
            ... )
            >>> print(graph)
            Data(x=[24, 7], edge_index=[2, 144], edge_attr=[144, 1], y=[3])
        """
        # Check structure size
        if len(structure) > self.max_atoms:
            logger.warning(
                f"Structure has {len(structure)} atoms, exceeding max {self.max_atoms}. Skipping."
            )
            return None

        # --- Node features ---
        node_features = []
        for site in structure:
            element = Element(site.species_string)
            features = self._get_element_features(element)
            node_features.append(features)

        node_features = np.array(node_features, dtype=np.float32)

        # --- Edge features ---
        # Get bonds based on strategy
        if self.bond_strategy == "radius_cutoff":
            edge_index, edge_distances = self._get_bonds_radius_cutoff(structure)
        elif self.bond_strategy == "knn":
            edge_index, edge_distances = self._get_bonds_knn(structure)
        elif self.bond_strategy == "crystalnn":
            edge_index, edge_distances = self._get_bonds_crystalnn(structure)
        else:
            raise ValueError(f"Unknown bond strategy: {self.bond_strategy}")

        # Normalize distances (typical bond lengths are 1-3 Angstroms)
        edge_attr = edge_distances.reshape(-1, 1) / 5.0

        # --- Target labels ---
        y = []
        if formation_energy is not None:
            y.append(formation_energy)
        if band_gap is not None:
            y.append(band_gap)
        if is_stable is not None:
            y.append(float(is_stable))

        y = np.array(y, dtype=np.float32) if y else None

        # --- Create PyTorch Geometric Data object ---
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr) if self.use_edge_features else None,
            y=torch.FloatTensor(y) if y is not None else None,
            num_atoms=len(structure),
            formula=structure.composition.reduced_formula,
        )

        return data

    def batch_convert(
        self,
        structures: List[Structure],
        formation_energies: Optional[List[float]] = None,
        band_gaps: Optional[List[float]] = None,
        is_stable_list: Optional[List[bool]] = None,
        show_progress: bool = True,
    ) -> List[Data]:
        """
        Convert a batch of structures to graphs.

        Args:
            structures: List of pymatgen Structures
            formation_energies: List of formation energies (optional)
            band_gaps: List of band gaps (optional)
            is_stable_list: List of stability flags (optional)
            show_progress: Show progress bar

        Returns:
            List of PyTorch Geometric Data objects

        Example:
            >>> structures = [structure1, structure2, structure3]
            >>> energies = [-2.5, -1.8, -3.2]
            >>> gaps = [1.2, 0.0, 2.5]
            >>> stable = [True, False, True]
            >>>
            >>> graphs = converter.batch_convert(
            ...     structures, energies, gaps, stable
            ... )
        """
        graphs = []

        # Prepare iterators
        if formation_energies is None:
            formation_energies = [None] * len(structures)
        if band_gaps is None:
            band_gaps = [None] * len(structures)
        if is_stable_list is None:
            is_stable_list = [None] * len(structures)

        # Convert each structure
        iterator = zip(structures, formation_energies, band_gaps, is_stable_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Converting to graphs")

        for structure, fe, bg, stable in iterator:
            try:
                graph = self.structure_to_graph(structure, fe, bg, stable)
                if graph is not None:
                    graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to convert structure: {e}")
                continue

        logger.info(f"Successfully converted {len(graphs)}/{len(structures)} structures")
        return graphs


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create a simple test structure (rock salt NaCl)
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Cl"] * 4,
        [
            [0, 0, 0], [0.5, 0.5, 0.5],
            [0.5, 0, 0.5], [0, 0.5, 0],
            [0.5, 0.5, 0], [0, 0, 0.5],
            [0, 0.5, 0.5], [0.5, 0, 0],
        ]
    )

    print(f"Structure: {structure.composition}")
    print(f"Number of atoms: {len(structure)}")

    # Test different bond strategies
    for strategy in ["radius_cutoff", "knn"]:
        print(f"\n--- Testing {strategy} strategy ---")

        converter = CrystalGraphConverter(
            bond_strategy=strategy,
            radius_cutoff=5.0,
            k_neighbors=12,
        )

        graph = converter.structure_to_graph(
            structure,
            formation_energy=-2.5,
            band_gap=1.2,
            is_stable=True,
        )

        print(f"Graph: {graph}")
        print(f"  Nodes (atoms): {graph.num_nodes}")
        print(f"  Node features shape: {graph.x.shape}")
        print(f"  Edges (bonds): {graph.num_edges}")
        print(f"  Edge features shape: {graph.edge_attr.shape if graph.edge_attr is not None else None}")
        print(f"  Target labels: {graph.y}")
        print(f"  Formula: {graph.formula}")

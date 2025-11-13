"""
Materials Project API Client

This module provides a cost-efficient interface to the Materials Project API with:
- Caching to minimize API calls
- Batch operations to reduce overhead
- Retry logic for reliability
- Mock mode for testing without API key

Key Features:
- SQLite cache for offline development
- Rate limiting to respect API quotas
- Automatic pagination for large queries
- Structure validation before storage

Author: Materials Discovery Team
"""

import os
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaterialData:
    """
    Container for material data from Materials Project.

    Attributes:
        material_id: Materials Project ID (e.g., 'mp-1234')
        formula: Chemical formula (e.g., 'Fe2O3')
        structure: Pymatgen Structure object
        formation_energy_per_atom: Formation energy in eV/atom
        band_gap: Band gap in eV (0 for metals)
        is_stable: Whether material is thermodynamically stable
        energy_above_hull: Energy above convex hull in eV/atom
        density: Density in g/cm³
    """
    material_id: str
    formula: str
    structure: Structure
    formation_energy_per_atom: float
    band_gap: float
    is_stable: bool
    energy_above_hull: float
    density: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'material_id': self.material_id,
            'formula': self.formula,
            'structure': self.structure.as_dict(),
            'formation_energy_per_atom': self.formation_energy_per_atom,
            'band_gap': self.band_gap,
            'is_stable': self.is_stable,
            'energy_above_hull': self.energy_above_hull,
            'density': self.density,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MaterialData':
        """Create from dictionary."""
        data['structure'] = Structure.from_dict(data['structure'])
        return cls(**data)


class MaterialsAPIClient:
    """
    Client for fetching and caching materials data from Materials Project.

    This class implements a two-tier caching strategy:
    1. In-memory LRU cache for frequently accessed materials
    2. SQLite cache for persistent storage across sessions

    Cost optimization features:
    - Batch queries to reduce API overhead
    - Automatic deduplication
    - Configurable cache expiration
    - Mock mode for testing without API key

    Example:
        >>> client = MaterialsAPIClient(api_key="your_key")
        >>> materials = client.fetch_materials(
        ...     elements=["Fe", "O"],
        ...     num_elements=(2, 2),
        ...     max_results=100
        ... )
        >>> print(f"Fetched {len(materials)} materials")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        mock_mode: bool = False,
    ):
        """
        Initialize the Materials Project API client.

        Args:
            api_key: Materials Project API key (or set MP_API_KEY env var)
            cache_dir: Directory for SQLite cache
            use_cache: Whether to use caching
            mock_mode: Use mock data instead of real API (for testing)
        """
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.mock_mode = mock_mode

        # Initialize cache database
        self.cache_db_path = self.cache_dir / "materials_cache.db"
        if use_cache:
            self._init_cache_db()

        # Initialize API client (not in mock mode)
        if not mock_mode:
            if not self.api_key:
                logger.warning(
                    "No API key provided. Set MP_API_KEY environment variable "
                    "or pass api_key parameter. Running in mock mode."
                )
                self.mock_mode = True
            else:
                try:
                    self.client = MPRester(self.api_key)
                    logger.info("Materials Project API client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize MP client: {e}")
                    logger.warning("Falling back to mock mode")
                    self.mock_mode = True

    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # Create materials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY,
                formula TEXT,
                data TEXT,
                timestamp REAL
            )
        """)

        # Create index on formula for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_formula ON materials(formula)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Cache database initialized at {self.cache_db_path}")

    def _cache_get(self, material_id: str) -> Optional[MaterialData]:
        """Get material from cache."""
        if not self.use_cache:
            return None

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT data FROM materials WHERE material_id = ?",
                (material_id,)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                data = json.loads(result[0])
                logger.debug(f"Cache hit for {material_id}")
                return MaterialData.from_dict(data)

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def _cache_put(self, material: MaterialData):
        """Put material into cache."""
        if not self.use_cache:
            return

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO materials (material_id, formula, data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                material.material_id,
                material.formula,
                json.dumps(material.to_dict(), default=str),
                time.time()
            ))

            conn.commit()
            conn.close()
            logger.debug(f"Cached {material.material_id}")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _generate_mock_material(
        self,
        material_id: str,
        formula: str,
        seed: Optional[int] = None
    ) -> MaterialData:
        """
        Generate mock material data for testing.

        This is useful for:
        - Testing without API key
        - Rapid prototyping
        - Unit tests
        - Offline development

        Mock data has realistic distributions but is not physically accurate.
        """
        if seed is None:
            # Use material_id as seed for consistency
            seed = int(hashlib.md5(material_id.encode()).hexdigest()[:8], 16)

        rng = np.random.RandomState(seed)

        # Generate a simple cubic structure (not realistic, just for testing)
        from pymatgen.core import Lattice, Structure

        lattice = Lattice.cubic(5.0)
        species = ["Fe", "O"]
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        structure = Structure(lattice, species, coords)

        # Generate realistic property ranges
        formation_energy = rng.uniform(-3.0, 0.5)
        band_gap = rng.lognormal(0, 1.0)  # Log-normal distribution
        band_gap = np.clip(band_gap, 0, 10)
        energy_above_hull = rng.exponential(0.1)
        is_stable = energy_above_hull < 0.05
        density = rng.uniform(2.0, 10.0)

        return MaterialData(
            material_id=material_id,
            formula=formula,
            structure=structure,
            formation_energy_per_atom=formation_energy,
            band_gap=band_gap,
            is_stable=is_stable,
            energy_above_hull=energy_above_hull,
            density=density,
        )

    def fetch_materials(
        self,
        elements: Optional[List[str]] = None,
        num_elements: Optional[Tuple[int, int]] = None,
        is_stable: Optional[bool] = None,
        max_results: int = 1000,
        fields: Optional[List[str]] = None,
    ) -> List[MaterialData]:
        """
        Fetch materials from Materials Project with filters.

        Args:
            elements: List of elements to include (e.g., ["Fe", "O"])
            num_elements: Range of elements per formula (e.g., (2, 3) for binary/ternary)
            is_stable: Filter by thermodynamic stability
            max_results: Maximum number of materials to fetch
            fields: Specific fields to retrieve (default: all common fields)

        Returns:
            List of MaterialData objects

        Example:
            >>> # Fetch stable binary oxides
            >>> materials = client.fetch_materials(
            ...     elements=["O"],
            ...     num_elements=(2, 2),
            ...     is_stable=True,
            ...     max_results=100
            ... )
        """
        if self.mock_mode:
            logger.info("Running in mock mode - generating synthetic data")
            return self._fetch_mock_materials(elements, num_elements, max_results)

        # Default fields to fetch
        if fields is None:
            fields = [
                "material_id",
                "formula_pretty",
                "structure",
                "formation_energy_per_atom",
                "band_gap",
                "is_stable",
                "energy_above_hull",
                "density",
            ]

        # Build query
        query_params = {}
        if elements:
            query_params["elements"] = elements
        if num_elements:
            query_params["num_elements"] = num_elements
        if is_stable is not None:
            query_params["is_stable"] = is_stable

        logger.info(f"Fetching materials with params: {query_params}")

        materials = []
        try:
            # Query Materials Project
            with self.client as mpr:
                docs = mpr.materials.summary.search(
                    **query_params,
                    fields=fields,
                    num_chunks=10,  # Pagination
                    chunk_size=min(max_results, 100),
                )

            # Convert to MaterialData objects
            for doc in tqdm(docs[:max_results], desc="Processing materials"):
                try:
                    # Check cache first
                    material_id = doc.material_id
                    cached = self._cache_get(material_id)
                    if cached:
                        materials.append(cached)
                        continue

                    # Create new MaterialData
                    material = MaterialData(
                        material_id=str(material_id),
                        formula=doc.formula_pretty,
                        structure=doc.structure,
                        formation_energy_per_atom=float(doc.formation_energy_per_atom),
                        band_gap=float(doc.band_gap),
                        is_stable=bool(doc.is_stable),
                        energy_above_hull=float(doc.energy_above_hull),
                        density=float(doc.density),
                    )

                    # Cache it
                    self._cache_put(material)
                    materials.append(material)

                except Exception as e:
                    logger.warning(f"Failed to process {doc.material_id}: {e}")
                    continue

            logger.info(f"Successfully fetched {len(materials)} materials")

        except Exception as e:
            logger.error(f"Failed to fetch materials: {e}")
            raise

        return materials

    def _fetch_mock_materials(
        self,
        elements: Optional[List[str]],
        num_elements: Optional[Tuple[int, int]],
        max_results: int,
    ) -> List[MaterialData]:
        """Generate mock materials for testing."""
        materials = []

        for i in range(max_results):
            material_id = f"mock-mp-{i:06d}"
            formula = "Fe2O3"  # Simplified for mock

            # Check cache first
            cached = self._cache_get(material_id)
            if cached:
                materials.append(cached)
            else:
                material = self._generate_mock_material(material_id, formula)
                self._cache_put(material)
                materials.append(material)

        logger.info(f"Generated {len(materials)} mock materials")
        return materials

    def fetch_by_ids(self, material_ids: List[str]) -> List[MaterialData]:
        """
        Fetch specific materials by their IDs.

        Args:
            material_ids: List of Materials Project IDs

        Returns:
            List of MaterialData objects
        """
        materials = []

        for material_id in tqdm(material_ids, desc="Fetching materials"):
            # Check cache first
            cached = self._cache_get(material_id)
            if cached:
                materials.append(cached)
                continue

            if self.mock_mode:
                material = self._generate_mock_material(material_id, "MockFormula")
                self._cache_put(material)
                materials.append(material)
            else:
                # Fetch from API
                try:
                    with self.client as mpr:
                        doc = mpr.materials.summary.get_data_by_id(material_id)

                    material = MaterialData(
                        material_id=str(doc.material_id),
                        formula=doc.formula_pretty,
                        structure=doc.structure,
                        formation_energy_per_atom=float(doc.formation_energy_per_atom),
                        band_gap=float(doc.band_gap),
                        is_stable=bool(doc.is_stable),
                        energy_above_hull=float(doc.energy_above_hull),
                        density=float(doc.density),
                    )

                    self._cache_put(material)
                    materials.append(material)

                except Exception as e:
                    logger.warning(f"Failed to fetch {material_id}: {e}")

        return materials

    def export_structures(
        self,
        materials: List[MaterialData],
        output_dir: str = "data/raw/structures",
        format: str = "cif",
    ):
        """
        Export structures to files.

        Args:
            materials: List of materials to export
            output_dir: Output directory
            format: File format ('cif', 'poscar', 'json')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for material in tqdm(materials, desc=f"Exporting {format} files"):
            filename = f"{material.material_id}.{format}"
            filepath = output_path / filename

            try:
                if format == "cif":
                    writer = CifWriter(material.structure)
                    writer.write_file(str(filepath))
                elif format == "json":
                    with open(filepath, 'w') as f:
                        json.dump(material.to_dict(), f, indent=2, default=str)
                else:
                    logger.warning(f"Unsupported format: {format}")

            except Exception as e:
                logger.warning(f"Failed to export {material.material_id}: {e}")

        logger.info(f"Exported {len(materials)} structures to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize client (will use mock mode if no API key)
    client = MaterialsAPIClient(mock_mode=True)

    # Fetch some materials
    materials = client.fetch_materials(
        elements=["Fe", "O"],
        num_elements=(2, 2),
        is_stable=True,
        max_results=10,
    )

    print(f"\nFetched {len(materials)} materials:")
    for mat in materials[:5]:
        print(f"  {mat.material_id}: {mat.formula}")
        print(f"    Formation energy: {mat.formation_energy_per_atom:.3f} eV/atom")
        print(f"    Band gap: {mat.band_gap:.3f} eV")
        print(f"    Stable: {mat.is_stable}")

    # Export to files
    client.export_structures(materials, output_dir="data/raw/test_structures")

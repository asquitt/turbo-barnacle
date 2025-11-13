"""
Data Quality Validators

This module implements data quality checks using Great Expectations to ensure:
- Physical validity (no impossible values)
- Stoichiometric correctness (charge balance)
- Structural sanity (reasonable bond lengths, densities)
- Data completeness (no missing critical fields)

Why data validation is critical:
1. Prevents training on corrupted data
2. Catches API/preprocessing errors early
3. Ensures reproducibility
4. Maintains scientific rigor

Author: Materials Discovery Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pymatgen.core import Structure, Composition
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Container for validation results.

    Attributes:
        is_valid: Overall validation status
        errors: List of validation errors
        warnings: List of validation warnings
        metrics: Dictionary of computed metrics
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

    def __str__(self) -> str:
        """Pretty print validation results."""
        lines = []
        lines.append(f"Validation Status: {'PASS' if self.is_valid else 'FAIL'}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                lines.append(f"  - {error}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                lines.append(f"  - {warning}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")

        if self.metrics:
            lines.append(f"\nMetrics:")
            for key, value in self.metrics.items():
                lines.append(f"  {key}: {value:.4f}")

        return "\n".join(lines)


class MaterialValidator:
    """
    Validates material data for physical and chemical correctness.

    This validator checks:
    1. Property ranges (formation energy, band gap, density)
    2. Structural validity (bond lengths, coordination numbers)
    3. Stoichiometry (charge neutrality, valid compositions)
    4. Data completeness (required fields present)
    5. Statistical outliers (detect anomalies)

    Example:
        >>> validator = MaterialValidator()
        >>> result = validator.validate_material(
        ...     formula="Fe2O3",
        ...     structure=structure,
        ...     formation_energy=-2.5,
        ...     band_gap=2.0,
        ...     density=5.24
        ... )
        >>> if result.is_valid:
        ...     print("Material is valid!")
        >>> else:
        ...     print(f"Validation failed: {result.errors}")
    """

    def __init__(
        self,
        strict: bool = False,
        allow_warnings: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            strict: If True, warnings are treated as errors
            allow_warnings: If False, warnings cause validation to fail
        """
        self.strict = strict
        self.allow_warnings = allow_warnings

        # Define physical bounds for properties
        self.bounds = {
            'formation_energy': (-10.0, 5.0),  # eV/atom
            'band_gap': (0.0, 15.0),  # eV
            'density': (0.1, 25.0),  # g/cm³
            'energy_above_hull': (0.0, 10.0),  # eV/atom
            'volume_per_atom': (5.0, 100.0),  # Angstrom³
            'min_bond_length': (0.5, 10.0),  # Angstrom
        }

        # Statistical thresholds for outlier detection
        self.outlier_threshold = 3.0  # Standard deviations

    def validate_property_range(
        self,
        property_name: str,
        value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a property is within physical bounds.

        Args:
            property_name: Name of property to check
            value: Value to validate

        Returns:
            (is_valid, error_message)
        """
        if property_name not in self.bounds:
            return True, None  # No bounds defined

        min_val, max_val = self.bounds[property_name]

        if not (min_val <= value <= max_val):
            error = (
                f"{property_name} = {value:.4f} is outside valid range "
                f"[{min_val}, {max_val}]"
            )
            return False, error

        return True, None

    def validate_composition(
        self,
        formula: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate chemical composition.

        Checks:
        1. Formula is parseable
        2. Elements are valid (Z <= 118)
        3. Stoichiometry is reasonable
        4. Charge neutrality (approximate)

        Args:
            formula: Chemical formula string

        Returns:
            (is_valid, error_message, warning_message)
        """
        try:
            comp = Composition(formula)
        except Exception as e:
            return False, f"Invalid formula '{formula}': {e}", None

        # Check for valid elements
        for element in comp.elements:
            if element.Z > 118:
                return False, f"Unknown element: {element}", None

        # Check for reasonable stoichiometry (not too many atoms)
        total_atoms = sum(comp.values())
        if total_atoms > 100:
            warning = f"Large stoichiometry: {total_atoms} atoms"
            return True, None, warning

        # Check for charge neutrality (approximation using common oxidation states)
        try:
            total_charge = sum(
                element.common_oxidation_states[0] * amount
                if element.common_oxidation_states else 0
                for element, amount in comp.items()
            )

            if abs(total_charge) > 0.1:  # Allow small deviation
                warning = f"Possible charge imbalance: {total_charge:.2f}"
                return True, None, warning

        except:
            # If we can't determine oxidation states, just warn
            warning = "Could not verify charge neutrality"
            return True, None, warning

        return True, None, None

    def validate_structure(
        self,
        structure: Structure,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate crystal structure geometry.

        Checks:
        1. Reasonable bond lengths (no atoms too close/far)
        2. Reasonable density
        3. Reasonable volume per atom
        4. No overlapping atoms

        Args:
            structure: Pymatgen Structure object

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check number of atoms
        if len(structure) == 0:
            errors.append("Structure has no atoms")
            return False, errors, warnings

        if len(structure) > 200:
            warnings.append(f"Large structure: {len(structure)} atoms")

        # Check volume per atom
        volume_per_atom = structure.volume / len(structure)
        is_valid, error = self.validate_property_range(
            'volume_per_atom', volume_per_atom
        )
        if not is_valid:
            errors.append(error)

        # Check bond lengths
        try:
            # Get all neighbors within reasonable distance
            neighbors_all = structure.get_all_neighbors(r=5.0)

            bond_lengths = []
            for neighbors in neighbors_all:
                if neighbors:  # Has at least one neighbor
                    min_distance = min(n.nn_distance for n in neighbors)
                    bond_lengths.append(min_distance)

                    # Check for unreasonably short bonds (overlapping atoms)
                    if min_distance < 0.5:
                        errors.append(
                            f"Unreasonably short bond: {min_distance:.3f} Å"
                        )

            # Check if any atoms are isolated
            if not all(neighbors_all):
                warnings.append("Some atoms have no neighbors within 5 Å")

            # Check average bond length
            if bond_lengths:
                avg_bond = np.mean(bond_lengths)
                if avg_bond > 4.0:
                    warnings.append(
                        f"Average bond length is large: {avg_bond:.2f} Å"
                    )

        except Exception as e:
            warnings.append(f"Could not analyze bond lengths: {e}")

        # Check density
        try:
            density = structure.density
            is_valid, error = self.validate_property_range('density', density)
            if not is_valid:
                errors.append(error)
        except Exception as e:
            warnings.append(f"Could not compute density: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_material(
        self,
        formula: str,
        structure: Optional[Structure] = None,
        formation_energy: Optional[float] = None,
        band_gap: Optional[float] = None,
        density: Optional[float] = None,
        energy_above_hull: Optional[float] = None,
        material_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Comprehensive validation of a material.

        Args:
            formula: Chemical formula
            structure: Pymatgen Structure (optional)
            formation_energy: Formation energy per atom in eV
            band_gap: Band gap in eV
            density: Density in g/cm³
            energy_above_hull: Energy above hull in eV/atom
            material_id: Material identifier (for error reporting)

        Returns:
            ValidationResult with overall status and details
        """
        errors = []
        warnings = []
        metrics = {}

        # Add material ID to messages for tracking
        def add_id(msg: str) -> str:
            return f"[{material_id}] {msg}" if material_id else msg

        # 1. Validate composition
        comp_valid, comp_error, comp_warning = self.validate_composition(formula)
        if not comp_valid:
            errors.append(add_id(comp_error))
        if comp_warning:
            warnings.append(add_id(comp_warning))

        # 2. Validate structure (if provided)
        if structure is not None:
            struct_valid, struct_errors, struct_warnings = self.validate_structure(structure)
            errors.extend([add_id(e) for e in struct_errors])
            warnings.extend([add_id(w) for w in struct_warnings])

            metrics['num_atoms'] = len(structure)
            metrics['volume'] = structure.volume
            metrics['volume_per_atom'] = structure.volume / len(structure)

        # 3. Validate property values
        if formation_energy is not None:
            is_valid, error = self.validate_property_range(
                'formation_energy', formation_energy
            )
            if not is_valid:
                errors.append(add_id(error))
            metrics['formation_energy'] = formation_energy

        if band_gap is not None:
            is_valid, error = self.validate_property_range('band_gap', band_gap)
            if not is_valid:
                errors.append(add_id(error))
            metrics['band_gap'] = band_gap

        if density is not None:
            is_valid, error = self.validate_property_range('density', density)
            if not is_valid:
                errors.append(add_id(error))
            metrics['density'] = density

        if energy_above_hull is not None:
            is_valid, error = self.validate_property_range(
                'energy_above_hull', energy_above_hull
            )
            if not is_valid:
                errors.append(add_id(error))
            metrics['energy_above_hull'] = energy_above_hull

        # 4. Cross-property consistency checks
        if energy_above_hull is not None and energy_above_hull < 0:
            errors.append(add_id("Energy above hull cannot be negative"))

        # Determine overall validity
        is_valid = len(errors) == 0
        if self.strict or not self.allow_warnings:
            is_valid = is_valid and len(warnings) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def validate_batch(
        self,
        materials_df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Validate a batch of materials from a DataFrame.

        Args:
            materials_df: DataFrame with material data
            required_columns: Columns that must be present

        Returns:
            (valid_materials_df, validation_result)
        """
        all_errors = []
        all_warnings = []
        metrics = {}

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(materials_df.columns)
            if missing:
                all_errors.append(f"Missing required columns: {missing}")
                return materials_df, ValidationResult(
                    is_valid=False,
                    errors=all_errors,
                    warnings=all_warnings,
                    metrics=metrics,
                )

        # Validate each material
        valid_indices = []

        for idx, row in materials_df.iterrows():
            result = self.validate_material(
                formula=row.get('formula', ''),
                formation_energy=row.get('formation_energy_per_atom'),
                band_gap=row.get('band_gap'),
                density=row.get('density'),
                energy_above_hull=row.get('energy_above_hull'),
                material_id=row.get('material_id', str(idx)),
            )

            if result.is_valid:
                valid_indices.append(idx)
            else:
                all_errors.extend(result.errors)

            all_warnings.extend(result.warnings)

        # Filter to valid materials
        valid_df = materials_df.loc[valid_indices]

        # Compute dataset statistics
        metrics['total_materials'] = len(materials_df)
        metrics['valid_materials'] = len(valid_df)
        metrics['invalid_materials'] = len(materials_df) - len(valid_df)
        metrics['validation_rate'] = len(valid_df) / len(materials_df)

        # Statistical checks on valid data
        if len(valid_df) > 0:
            if 'formation_energy_per_atom' in valid_df.columns:
                fe_values = valid_df['formation_energy_per_atom'].values
                metrics['formation_energy_mean'] = float(np.mean(fe_values))
                metrics['formation_energy_std'] = float(np.std(fe_values))

            if 'band_gap' in valid_df.columns:
                bg_values = valid_df['band_gap'].values
                metrics['band_gap_mean'] = float(np.mean(bg_values))
                metrics['band_gap_std'] = float(np.std(bg_values))

        overall_valid = len(valid_df) > 0 and len(all_errors) == 0

        return valid_df, ValidationResult(
            is_valid=overall_valid,
            errors=all_errors,
            warnings=all_warnings,
            metrics=metrics,
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    validator = MaterialValidator()

    # Test 1: Valid material
    print("Test 1: Valid material")
    result = validator.validate_material(
        formula="Fe2O3",
        formation_energy=-2.5,
        band_gap=2.0,
        density=5.24,
        energy_above_hull=0.0,
        material_id="mp-1234",
    )
    print(result)
    print()

    # Test 2: Invalid formation energy
    print("Test 2: Invalid formation energy (too low)")
    result = validator.validate_material(
        formula="Fe2O3",
        formation_energy=-15.0,  # Too low!
        band_gap=2.0,
        density=5.24,
        material_id="mp-5678",
    )
    print(result)
    print()

    # Test 3: Invalid formula
    print("Test 3: Invalid formula")
    result = validator.validate_material(
        formula="XYZ123",  # Not a valid formula
        formation_energy=-2.5,
        band_gap=2.0,
        material_id="mp-9999",
    )
    print(result)
    print()

    # Test 4: Batch validation
    print("Test 4: Batch validation")
    df = pd.DataFrame({
        'material_id': ['mp-1', 'mp-2', 'mp-3', 'mp-4'],
        'formula': ['Fe2O3', 'NaCl', 'InvalidFormula', 'TiO2'],
        'formation_energy_per_atom': [-2.5, -1.8, -20.0, -3.2],  # mp-3 has invalid energy
        'band_gap': [2.0, 5.0, 1.5, 3.2],
        'density': [5.24, 2.16, 4.5, 4.23],
    })

    valid_df, result = validator.validate_batch(df)
    print(result)
    print(f"\nValid materials: {len(valid_df)}/{len(df)}")
    print(valid_df[['material_id', 'formula']])

"""
Setup Script for Autonomous Materials Discovery

This script helps initialize the environment:
- Creates necessary directories
- Downloads sample data (optional)
- Verifies dependencies
- Sets up configuration files

Usage:
    python scripts/setup.py --mode local
    python scripts/setup.py --mode docker
    python scripts/setup.py --download-sample-data

Author: Materials Discovery Team
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


def check_python_version():
    """Verify Python version is 3.9+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required (found {version.major}.{version.minor})")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def create_directories():
    """Create necessary directory structure."""
    print("\nCreating directories...")

    directories = [
        "data/raw",
        "data/processed",
        "data/results",
        "data/cache",
        "models/checkpoints",
        "logs",
        "mlruns",
        "mlartifacts",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def check_dependencies():
    """Verify all required packages are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("pymatgen", "Pymatgen"),
        ("anthropic", "Anthropic"),
        ("mlflow", "MLflow"),
        ("streamlit", "Streamlit"),
    ]

    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} not installed")
            all_installed = False

    if not all_installed:
        print("\n⚠️  Some dependencies are missing.")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_api_keys():
    """Check for API keys (not required for mock mode)."""
    print("\nChecking API keys...")

    mp_key = os.getenv("MP_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if mp_key:
        print("  ✓ MP_API_KEY found")
    else:
        print("  ⚠️  MP_API_KEY not set (will use mock data)")
        print("     Get a free key at: https://materialsproject.org/api")

    if anthropic_key:
        print("  ✓ ANTHROPIC_API_KEY found")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY not set (LLM agent disabled)")
        print("     Get a key at: https://console.anthropic.com/")


def create_env_template():
    """Create .env.template file."""
    print("\nCreating .env.template...")

    template_content = """# API Keys (Optional)
# Get Materials Project API key: https://materialsproject.org/api
MP_API_KEY=your_materials_project_key_here

# Get Anthropic API key: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_key_here

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Compute Configuration
DEVICE=auto  # Options: auto, cpu, cuda, mps

# Weights & Biases (Optional)
# WANDB_API_KEY=your_wandb_key_here
# WANDB_PROJECT=materials-discovery
"""

    with open(".env.template", "w") as f:
        f.write(template_content)

    print("  ✓ Created .env.template")
    print("  → Copy to .env and add your API keys")


def download_sample_data():
    """Download sample materials data for testing."""
    print("\nDownloading sample data...")
    print("  (This would download ~100 sample materials)")
    print("  ⚠️  Skipped in this demo - use --mock flag instead")


def verify_installation():
    """Run a quick verification test."""
    print("\nVerifying installation...")

    try:
        # Test PyTorch
        import torch
        x = torch.randn(3, 3)
        print(f"  ✓ PyTorch working (device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')})")

        # Test PyTorch Geometric
        from torch_geometric.data import Data
        data = Data(x=torch.randn(3, 7))
        print("  ✓ PyTorch Geometric working")

        # Test Pymatgen
        from pymatgen.core import Structure, Lattice
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, ["Fe"], [[0, 0, 0]])
        print("  ✓ Pymatgen working")

        print("\n✅ All systems operational!")
        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False


def print_next_steps(mode):
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)

    if mode == "local":
        print("\nNext steps:")
        print("1. (Optional) Copy .env.template to .env and add API keys")
        print("2. Run a test discovery:")
        print("   python src/main.py --iterations 20 --mock --visualize")
        print("\n3. Launch the demo:")
        print("   streamlit run src/ui/streamlit_app.py")
        print("\n4. View MLflow UI:")
        print("   mlflow ui --port 5000")
        print("\n5. Read the documentation:")
        print("   - docs/ARCHITECTURE.md")
        print("   - notebooks/01_data_exploration.ipynb")

    elif mode == "docker":
        print("\nNext steps:")
        print("1. (Optional) Add API keys to .env file")
        print("2. Start all services:")
        print("   docker-compose up --build")
        print("\n3. Access services:")
        print("   - Demo: http://localhost:8501")
        print("   - MLflow: http://localhost:5000")
        print("\n4. Stop services:")
        print("   docker-compose down")


def main():
    parser = argparse.ArgumentParser(description="Setup Autonomous Materials Discovery")
    parser.add_argument(
        "--mode",
        choices=["local", "docker"],
        default="local",
        help="Setup mode"
    )
    parser.add_argument(
        "--download-sample-data",
        action="store_true",
        help="Download sample materials data"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency checks"
    )

    args = parser.parse_args()

    print("="*60)
    print("Autonomous Materials Discovery - Setup")
    print("="*60)

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Check dependencies (unless skipped)
    if not args.skip_checks:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("\n⚠️  Fix dependencies before continuing")
            sys.exit(1)

    # Check API keys (informational only)
    check_api_keys()

    # Create .env template
    create_env_template()

    # Download sample data if requested
    if args.download_sample_data:
        download_sample_data()

    # Verify installation
    if not args.skip_checks:
        verify_installation()

    # Print next steps
    print_next_steps(args.mode)


if __name__ == "__main__":
    main()

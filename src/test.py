from pathlib import Path
import site
import sys


def _bootstrap_venv_packages() -> None:
    """Add project venv site-packages so imports work from project root."""
    project_root = Path(__file__).resolve().parent.parent
    lib_dir = project_root / "venv" / "lib"
    if not lib_dir.exists():
        return

    versioned_site = lib_dir / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if versioned_site.exists():
        site.addsitedir(str(versioned_site))
        return

    fallback = sorted(lib_dir.glob("python*/site-packages"))
    if fallback:
        site.addsitedir(str(fallback[0]))


_bootstrap_venv_packages()

from spatial_engine import analyze_location
from risk_engine import classify_risk
from ml_model import predict_risk_ml

lat = 12.9300
lon = 77.6700

# Step 1: Spatial
result = analyze_location(lat, lon)

# Step 2: Rule-based risk
rule_risk = classify_risk(result)

# Step 3: ML risk
ml_risk = predict_risk_ml(result)

print("\n=== SPATIAL RESULT ===")
print(result)

print("\n=== RULE-BASED RISK ===")
print(rule_risk)

print("\n=== ML MODEL RISK ===")
print(ml_risk)

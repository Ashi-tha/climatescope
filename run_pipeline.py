"""
ClimateScope — Master Pipeline Runner
Runs all steps in sequence.
Usage: python run_pipeline.py [--skip-download]
"""

import subprocess
import sys
from pathlib import Path


STEPS = [
    ("01_data_acquisition.py",  "📥  Step 1 — Data Acquisition"),
    ("02_data_exploration.py",  "🔍  Step 2 — Data Exploration"),
    ("03_data_cleaning.py",     "🧹  Step 3 — Data Cleaning & Preprocessing"),
    ("04_data_analysis.py",     "📊  Step 4 — Statistical Analysis"),
    ("05_visualizations.py",    "📈  Step 5 — Build Standalone Charts"),
]


def run_step(script: str, label: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable, f"src/{script}"], capture_output=False)
    if result.returncode != 0:
        print(f"\n  ❌  {script} failed (exit code {result.returncode})")
        return False
    return True


def main() -> None:
    skip_download = "--skip-download" in sys.argv

    print("\n" + "═"*60)
    print("  🌍  ClimateScope — Full Pipeline")
    print("═"*60)

    for script, label in STEPS:
        if skip_download and "acquisition" in script:
            print(f"\n  ⏭️   Skipping: {label}")
            continue
        if not run_step(script, label):
            print("\n  ⚠️   Pipeline halted. Fix the error above and re-run.")
            sys.exit(1)

    print("\n" + "═"*60)
    print("  ✅  Pipeline complete!")
    print("  🚀  Launch dashboard: streamlit run dashboard/app.py")
    print("  📂  Charts:           reports/charts/")
    print("  📋  Reports:          reports/")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()

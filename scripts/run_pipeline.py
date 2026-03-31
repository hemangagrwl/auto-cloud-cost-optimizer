import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "scripts/collector_realtime.py",
    "scripts/preprocess_realtime.py",
    "scripts/damp_stage.py",
    "scripts/decision_stage_v3.py",
]

def run_script(script_path: str):
    print(f"\n=== Running {script_path} ===")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Failed while running {script_path}")

def main():
    for script in SCRIPTS:
        run_script(script)

    print("\nPipeline completed successfully.")
    print("Latest handoff file: data/processed/final_output.csv")

if __name__ == "__main__":
    main()

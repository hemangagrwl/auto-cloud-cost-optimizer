import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent

SLEEP_SECONDS = 60
S3_EVERY_N_CYCLES = int(os.getenv("S3_EVERY_N_CYCLES", "60"))

def run_script(script_and_args: list[str]):
    script_path = script_and_args[0]
    script_args = script_and_args[1:]
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script_path), *script_args],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Failed while running {script_path}")

def main():
    cycle_no = 0
    while True:
        cycle_no += 1
        print(f"\n========== Pipeline cycle started at {datetime.now(timezone.utc).isoformat()} ==========")
        try:
            include_s3 = (cycle_no % max(S3_EVERY_N_CYCLES, 1) == 1)
            collector_cmd = ["scripts/collector_realtime.py"] if include_s3 else ["scripts/collector_realtime.py", "--skip-s3"]

            scripts = [
                collector_cmd,
                ["scripts/preprocess_realtime.py"],
                ["scripts/damp_stage.py"],
                ["scripts/decision_stage_v3.py"],
            ]

            print(
                f"Cycle #{cycle_no}: S3 collection {'enabled' if include_s3 else 'skipped'} "
                f"(configured every {S3_EVERY_N_CYCLES} cycles)"
            )

            for script in scripts:
                print(f"\n--- Running {' '.join(script)} ---")
                run_script(script)

            print("Cycle completed successfully.")
        except Exception as e:
            print(f"Pipeline cycle failed: {e}")

        print(f"Sleeping for {SLEEP_SECONDS} seconds...\n")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()

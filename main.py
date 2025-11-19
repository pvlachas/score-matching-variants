"""
exec(open("tutorial.py").read())
"""

# Entrypoint to run all demos if you like:
#   python main.py
#
# Or run them individually from the root directory:
#   python experiments/1_score_matching.py
#   python experiments/2_sliced_score_matching.py
#   python experiments/3_denoising_score_matching.py
#   python experiments/4_noise_conditional_score_networks.py

import subprocess
import sys
from pathlib import Path

def run(script):
    print(f"\n=== Running: {script} ===\n")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    experiments_dir = Path("experiments")
    run(str(experiments_dir / "1_score_matching.py"))
    run(str(experiments_dir / "2_sliced_score_matching.py"))
    run(str(experiments_dir / "3_denoising_score_matching.py"))
    run(str(experiments_dir / "4_noise_conditional_score_networks.py"))

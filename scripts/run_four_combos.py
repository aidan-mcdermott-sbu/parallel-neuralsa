import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def as_hydra_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def flatten_overrides(section, values):
    return [f"{section}.{key}={as_hydra_value(value)}" for key, value in values.items()]


def main():
    parser = argparse.ArgumentParser(description="Run the four pi/C learning combinations.")
    parser.add_argument(
        "--config",
        default="scripts/conf/four_combos_knapsack.yaml",
        help="Path to the four-combo YAML parameter file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    common = [f"+experiment={cfg['experiment']}"]
    common.extend(f"{key}={as_hydra_value(value)}" for key, value in cfg["problem"].items())
    common.extend(flatten_overrides("training", cfg["training"]))
    common.extend(flatten_overrides("sa", cfg["sa"]))
    common.append("hydra.job.chdir=True")

    output_root = cfg["output_root"]
    for combo in cfg["combos"]:
        print(f"=== RUN {combo['name']} ===", flush=True)
        overrides = common + [
            f"training.learn_policy={as_hydra_value(combo['learn_policy'])}",
            f"sa.learn_c={as_hydra_value(combo['learn_c'])}",
            f"hydra.run.dir={output_root}/{combo['name']}",
        ]
        subprocess.run(
            [sys.executable, "scripts/main.py", *overrides],
            cwd=repo_root,
            check=True,
        )


if __name__ == "__main__":
    main()

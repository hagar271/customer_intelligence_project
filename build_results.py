"""Builds every result file the dashboard needs from the raw Online Retail II dataset.

Run it by double-clicking build_results.bat (Windows) or:  python build_results.py [path/to/online_retail_II.xlsx] [--full]
It also pins the scikit-learn (and xgboost) version in requirements.txt, so the online app can load the saved model.
"""
import argparse
import re
import time
from pathlib import Path

import config
import pipeline

HERE = Path(__file__).resolve().parent


def choose_file(arg):
    if arg:
        return Path(arg)
    for pattern in ("online_retail_II*.xlsx", "online_retail_II*.xls", "online_retail_II*.csv", "*.xlsx"):
        found = sorted(HERE.glob(pattern))
        if found:
            return found[0]
    try:                                            # no file next to the script -> ask with a window
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(title="Choose online_retail_II.xlsx",
                                            filetypes=[("Dataset", "*.xlsx *.xls *.csv")])
        root.destroy()
        if chosen:
            return Path(chosen)
    except Exception:
        pass
    typed = input("Path of the dataset file (online_retail_II.xlsx): ").strip().strip('"')
    return Path(typed)


def pin_requirements(best_model: str):
    req = HERE / "requirements.txt"
    if not req.exists():
        return
    import sklearn
    text = req.read_text()
    text = re.sub(r"(?m)^scikit-learn.*$", f"scikit-learn=={sklearn.__version__}", text)
    if best_model == "XGBoost":
        import xgboost
        text = re.sub(r"(?m)^xgboost.*$", f"xgboost=={xgboost.__version__}", text)
    req.write_text(text)
    print(f"requirements.txt pinned: scikit-learn=={sklearn.__version__}"
          + (f", xgboost=={xgboost.__version__}" if best_model == "XGBoost" else ""))


def main():
    ap = argparse.ArgumentParser(description="Build the dashboard result files")
    ap.add_argument("data", nargs="?", help="path to online_retail_II.xlsx (or .csv)")
    ap.add_argument("--full", action="store_true", help="full grid search like the notebook (slower)")
    args = ap.parse_args()

    path = choose_file(args.data)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    print(f"Dataset : {path}\nOutput  : {config.ARTIFACTS_DIR}\nMode    : {'full grid' if args.full else 'fast'}\n")

    start = time.time()
    summary = pipeline.run_pipeline(str(path), config.ARTIFACTS_DIR, fast=not args.full, use_xgb=True,
                                    progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    pin_requirements(summary["best_model"])

    m, b = summary["test_metrics"], summary["baseline_metrics"]
    print(f"\nDone in {time.time() - start:.0f} s")
    print(f"Best model: {summary['best_model']} | F1 {m['F1']:.3f} (baseline {b['F1']:.3f}) | "
          f"Recall {m['Recall']:.3f} | ROC-AUC {m['ROC-AUC']:.3f}")

    big = [f for f in config.ARTIFACTS_DIR.iterdir() if f.is_file() and f.stat().st_size > 20 * 1024 * 1024]
    for f in big:
        print(f"WARNING: {f.name} is {f.stat().st_size / 1e6:.0f} MB - too big for a browser upload to GitHub.")
    print("\nNext: double-click publish_to_github.bat (or upload all the files in this folder to GitHub).")
    print(f"You can keep the dataset file ({path.name}) OUT of the upload - it is not needed online.")


if __name__ == "__main__":
    main()

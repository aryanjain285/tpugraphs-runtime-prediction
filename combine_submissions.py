"""
combine_submissions.py — Merge predictions from all 5 collections
into the final Kaggle submission CSV.

The Kaggle competition expects a CSV with columns:
  ID  — formatted as "tile:xla:<graph_id>" or "layout:<source>:<search>:<graph_id>"
  TopConfigs — semicolon-separated ranked config indices

This script reads the output from predict.py and also cross-references
with the sample_submission.csv to ensure all required IDs are present.

Usage:
    python combine_submissions.py
    python combine_submissions.py --sample_submission data/sample_submission.csv
"""

import os
import argparse
import pandas as pd

from config import SUBMISSION_DIR, ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Combine submission CSVs")
    parser.add_argument("--predictions", type=str,
                        default=os.path.join(SUBMISSION_DIR, "all_predictions.csv"),
                        help="Path to predictions CSV from predict.py")
    parser.add_argument("--sample_submission", type=str, default=None,
                        help="Path to sample_submission.csv for reference")
    parser.add_argument("--output", type=str,
                        default=os.path.join(SUBMISSION_DIR, "final_submission.csv"),
                        help="Path to output submission CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    print("Combining submission files...")

    if not os.path.exists(args.predictions):
        print(f"Error: {args.predictions} not found. Run predict.py first.")
        return

    pred_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(pred_df)} predictions")

    # If sample submission is provided, check coverage
    if args.sample_submission and os.path.exists(args.sample_submission):
        sample_df = pd.read_csv(args.sample_submission)
        required_ids = set(sample_df["ID"])
        predicted_ids = set(pred_df["ID"])

        missing = required_ids - predicted_ids
        if missing:
            print(f"Warning: {len(missing)} IDs in sample_submission not predicted:")
            for mid in list(missing)[:10]:
                print(f"  {mid}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

            # Fill missing with default (first config)
            missing_rows = []
            for mid in missing:
                missing_rows.append({"ID": mid, "TopConfigs": "0;1;2;3;4"})
            missing_df = pd.DataFrame(missing_rows)
            pred_df = pd.concat([pred_df, missing_df], ignore_index=True)
            print(f"Filled {len(missing)} missing predictions with defaults")

        extra = predicted_ids - required_ids
        if extra:
            print(f"Note: {len(extra)} predicted IDs not in sample_submission (OK)")

    # Save final submission
    pred_df.to_csv(args.output, index=False)
    print(f"\nFinal submission saved to: {args.output}")
    print(f"Total rows: {len(pred_df)}")
    print("\nNext step: upload to Kaggle:")
    print(f"  kaggle competitions submit -c predict-ai-model-runtime -f {args.output} -m 'GNN submission'")


if __name__ == "__main__":
    main()

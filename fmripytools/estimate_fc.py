import numpy as np
import argparse
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


def estimate_fc(input_file: str,
                output_file: str,
                method: str = "correlation") -> np.ndarray:
    """
    Estimate functional connectivity (FC) from a BOLD time series.

    Parameters
    ----------
    input_file : str
        Path to input CSV containing BOLD time series (time x nodes).
    output_file : str
        Path to save the FC matrix as a CSV.
    method : str, optional
        Method for FC estimation. Currently supports:
        - 'correlation' (default)

    Returns
    -------
    fc : np.ndarray
        Functional connectivity matrix (nodes x nodes).
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logging.info(f"Loading input BOLD data from {input_file}")
    bold_ts = np.loadtxt(input_file, delimiter=',').T  # shape: nodes x time

    n_nodes, n_timepoints = bold_ts.shape
    if n_timepoints <= n_nodes:
        logging.error(
            f"Timepoints ({n_timepoints}) must exceed nodes ({n_nodes}). "
            "Check data orientation or preprocessing."
        )
        sys.exit(1)

    logging.info(f"Estimating FC using method: {method}")
    if method == "correlation":
        fc = np.corrcoef(bold_ts)
    else:
        logging.error(f"Unknown method: {method}")
        sys.exit(1)

    logging.info(f"Saving FC matrix to {output_file}")
    np.savetxt(output_file, fc, delimiter=',')
    return fc


def main():
    parser = argparse.ArgumentParser(
        description="Estimate functional connectivity from BOLD data")
    parser.add_argument("--input", required=True, type=str,
                        help="Input CSV with BOLD time series")
    parser.add_argument("--output", required=True, type=str,
                        help="Output CSV for FC matrix")
    parser.add_argument(
        "--method", type=str, default="correlation",
        choices=["correlation"],  # extendable
        help="Functional connectivity estimation method"
    )
    args = parser.parse_args()

    estimate_fc(args.input, args.output, args.method)


if __name__ == "__main__":
    main()

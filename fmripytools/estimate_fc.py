import numpy as np
import argparse
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


def load_bold_data(input_files):
    """
    Load one or more BOLD time series files and concatenate along time.

    Parameters
    ----------
    input_files : list of str
        Paths to CSV files containing BOLD data (time x nodes).

    Returns
    -------
    bold_ts : np.ndarray
        Concatenated BOLD data (nodes x total_time).
    """
    all_data = []
    for f in input_files:
        if not os.path.exists(f):
            logging.error(f"Input file not found: {f}")
            sys.exit(1)
        logging.info(f"Loading {f}")
        data = np.loadtxt(f, delimiter=',').T  # nodes x time
        all_data.append(data)

    # check consistent node dimension
    n_nodes = {d.shape[0] for d in all_data}
    if len(n_nodes) != 1:
        logging.error("Input files have inconsistent number of nodes.")
        sys.exit(1)

    bold_ts = np.concatenate(all_data, axis=1)  # concatenate along time
    logging.info(f"Concatenated BOLD shape: {bold_ts.shape}")
    return bold_ts


def estimate_fc(input_files, output_file, method="correlation"):
    """
    Estimate functional connectivity (FC) from BOLD time series.

    Parameters
    ----------
    input_files : list of str
        Paths to CSV files with BOLD data (time x nodes).
    output_file : str
        Path to save the FC matrix as a CSV.
    method : str
        Method for FC estimation. Currently supports: 'correlation'.

    Returns
    -------
    fc : np.ndarray
        Functional connectivity matrix (nodes x nodes).
    """
    bold_ts = load_bold_data(input_files)
    n_nodes, n_timepoints = bold_ts.shape

    if n_timepoints <= n_nodes:
        logging.warning(
            f"Timepoints ({n_timepoints}) should exceed nodes ({n_nodes}). "
            "Check data orientation or preprocessing."
        )

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
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more input CSVs with BOLD time series"
    )
    parser.add_argument("--output", required=True,
                        help="Output CSV for FC matrix")
    parser.add_argument(
        "--method", default="correlation", choices=["correlation"],
        help="Functional connectivity estimation method"
    )
    args = parser.parse_args()

    estimate_fc(args.input, args.output, args.method)


if __name__ == "__main__":
    main()

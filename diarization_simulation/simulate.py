#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.special import gammaln
import random
from tqdm import tqdm
import argparse

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8 and below
    from importlib_resources import files
import os

import numpy as np
from numba import jit


@jit(nopython=True)
def simulate_sample(true_vocs, alpha, mu, tau, n_recs, n_speakers, distribution_type):
    """
    Numba-optimized function to simulate detections for a single sample.

    Args:
        true_vocs: (n_recs, n_speakers) array of true vocalization counts
        alpha: (n_speakers, n_speakers) array of alpha parameters
        mu: (n_speakers, n_speakers) array of mu parameters
        tau: scalar tau parameter
        n_recs: number of recordings
        n_speakers: number of speakers
        distribution_type: 0 for Poisson, 1 for Gamma

    Returns:
        detected_vocs: (n_recs, n_speakers) array of detected counts
    """
    detected_vocs = np.zeros((n_recs, n_speakers), dtype=np.int32)

    # Simulate detections for each recording
    for k in range(n_recs):
        for j in range(n_speakers):  # target speaker
            detected_vocs[k, j] = 0
            for i in range(n_speakers):  # source speaker
                # Beta-binomial simulation
                lambda_ = np.random.gamma(alpha[i, j], mu[i, j] / alpha[i, j])
                mean = lambda_ * true_vocs[k, i]

                if true_vocs[k, i] > 0 and mean > 0:
                    if distribution_type == 0:  # poisson
                        detected_vocs[k, j] += np.random.poisson(mean)
                    else:  # gamma
                        sd = np.sqrt(mean / tau)
                        shape = (mean / sd) ** 2
                        detected_vocs[k, j] += int(
                            np.random.gamma(shape, mean / shape) + 0.5
                        )

    return detected_vocs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth",
        required=True,
        help="Path to the synthetic truth dataset (in csv format)",
    )
    parser.add_argument("--output", required=True, help="Location of the output CSV")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples per observation"
    )
    parser.add_argument(
        "--average-hyperpriors",
        action="store_true",
        help="Use the mean value of the hyperpriors (mu and alpha)",
    )
    parser.add_argument(
        "--unique-hyperpriors",
        action="store_true",
        help="Use fixed hyperpriors (mu and alpha) throughout all samples",
    )
    parser.add_argument(
        "--algo", choices=["vtc", "lena"], required=True, help="Algorithm to simulate"
    )
    parser.add_argument(
        "--distribution",
        choices=["poisson", "gamma"],
        help="Distribution for vocalization counts. We propose two approximation schemes for the underlying distribution: a Poisson scheme (which inflates the actual variance a bit), and a gamma scheme (which tries to capture the correct variance, but is a poor approximation for small numbers.)",
    )

    args = parser.parse_args()

    speakers = ["CHI", "OCH", "FEM", "MAL"]

    samples_location = f"output/{args.algo}.npz"

    # Replace pkg_resources with importlib.resources
    try:
        data_path = files("diarization_simulation") / "data" / f"{args.algo}.npz"
        samples_location = str(data_path)
    except ImportError:
        # Fallback to relative path if package not found
        samples_location = f"data/{args.algo}.npz"

    samples = np.load(samples_location)

    # Load truth data
    recs = pd.read_csv(args.truth)

    assert set(speakers).issubset(
        recs.columns
    ), "Truth data must contain all speakers (CHI, OCH, FEM, MAL)"

    n_recs = len(recs)
    true_vocs = np.stack(recs[speakers].values)
    observation = recs["observation"].values

    # Initialize detection arrays
    detected = {speaker: [] for speaker in speakers}
    detected["observation"] = []
    detected["sample"] = []

    # Extract hyperparameters
    n_available_samples = samples["mus"].shape[0]

    mu_ = samples["mus"]
    alpha_ = samples["alphas"]
    tau_ = samples["tau"]

    mu = mu_.mean(axis=0)
    alpha = alpha_.mean(axis=0)
    tau = tau_.mean(axis=0)

    drawn = np.random.choice(np.arange(n_available_samples), args.samples)
    distribution_type = 0 if args.distribution == "poisson" else 1

    for n, s in tqdm(enumerate(drawn)):
        current_mu = mu if args.average_hyperpriors else mu_[s]
        current_alpha = alpha if args.average_hyperpriors else alpha_[s]
        current_tau = tau if args.average_hyperpriors else tau_[s]

        # Call the numba function
        sample_detections = simulate_sample(
            true_vocs,
            current_alpha,
            current_mu,
            current_tau,
            n_recs,
            len(speakers),
            distribution_type
        )

        # Store results
        for k in range(n_recs):
            for j, speaker in enumerate(speakers):
                detected[speaker].append(sample_detections[k, j])

            detected["observation"].append(observation[k])
            detected["sample"].append(n)

    # Save results
    detected = pd.DataFrame(detected)
    detected.sort_values(["sample", "observation"], inplace=True)
    detected.to_csv(args.output)


if __name__ == "__main__":
    main()

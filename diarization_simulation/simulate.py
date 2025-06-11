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
        default=True,
        help="Use the mean value of the hyperpriors (mu and alpha)",
    )
    parser.add_argument(
        "--unique-hyperpriors",
        action="store_true",
        help="Use fixed hyperpriors (mu and alpha) throughout all samples",
        default=True,
    )
    parser.add_argument(
        "--algo", choices=["vtc", "lena"], required=True, help="Algorithm to simulate"
    )
    parser.add_argument(
        "--distribution",
        choices=["poisson", "normal"],
        help="Distribution for vocalization counts. We propose to approximation schemes for the underlying distribution: a Poisson scheme (which inflates the actual variance a bit), a normal scheme (which tries to capture the correct variance, but is a poor approximation for small numbers.)",
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
    detected_vocs = np.zeros((args.samples, n_recs, len(speakers)))
    detected = {speaker: [] for speaker in speakers}
    detected["observation"] = []
    detected["sample"] = []

    # Extract hyperparameters
    n_available_samples = samples["mus"].shape[0]
    mu = samples["mus"].mean(axis=0)
    alpha = samples["alphas"].mean(axis=0)
    tau = samples["tau"].mean(axis=0)

    # Sample simulation
    drawn = np.random.choice(np.arange(n_available_samples), args.samples)
    for n, s in tqdm(enumerate(drawn)):
        # Update hyperparameters if not averaging
        if not args.average_hyperpriors:
            if args.unique_hyperpriors:
                s = drawn[0]

            mu = samples["mus"][s]
            eta = samples["alphas"][s]
            tau = samples["tau"][s]

        # Simulate detections for each recording
        for k in range(n_recs):
            for j, speaker in enumerate(speakers):
                detected_vocs[n, k, j] = 0
                for i, true_speaker in enumerate(speakers):
                    # Beta-binomial simulation
                    lambda_ = np.random.gamma(alpha[i, j], mu[i, j] / alpha[i, j])
                    mean = lambda_ * true_vocs[k, i]

                    if args.distribution == "poisson":
                        detected_vocs[n, k, j] += np.random.poisson(mean)
                    else:
                        sd = np.sqrt(mean / tau)
                        detected_vocs[n, k, j] += int(np.random.normal(mean, sd) + 0.5)

                detected[speaker].append(detected_vocs[n, k, j])

            detected["observation"].append(observation[k])
            detected["sample"].append(n)

    # Save results
    detected = pd.DataFrame(detected)
    detected.sort_values(["sample", "observation"], inplace=True)
    detected.to_csv(args.output)


if __name__ == "__main__":
    main()

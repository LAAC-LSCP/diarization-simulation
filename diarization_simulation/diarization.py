import pandas as pd
import numpy as np
from scipy.special import gammaln
import random
from tqdm import tqdm
import argparse
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path

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
def simulate_sample(
        true_vocs, alpha, mu, tau, n_recs, n_speakers, distribution_type,
):
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
                lambda_ = np.random.gamma(alpha[i, j], mu[i, j] / alpha[i, j])
                mean = lambda_ * true_vocs[k, i]

                if true_vocs[k, i] > 0 and mean > 0:
                    if distribution_type == 0:  # poisson
                        detected_vocs[k, j] += np.random.poisson(mean)
                    else:  # gamma
                        sd = np.sqrt(mean / tau)
                        shape = (mean / sd) ** 2
                        detected_vocs[k, j] += int(
                            np.random.gamma(shape, mean / shape) + 0.5,
                        )

    return detected_vocs


class DiarizationSimulator:
    """
    Main class for diarization simulation.

    This class provides methods to simulate speaker diarization detection
    based on true vocalization counts.
    """

    def __init__(self, algorithm: str = "vtc", distribution: str = "poisson"):
        """
        Initialize the simulator.

        Args:
            algorithm: Algorithm to simulate ("vtc" or "lena")
            distribution: Distribution type ("poisson" or "gamma")
        """
        self.algorithm = algorithm.lower()
        self.distribution = distribution.lower()
        self.speakers = ["CHI", "OCH", "FEM", "MAL"]

        if self.algorithm not in ["vtc", "lena"]:
            raise ValueError("Algorithm must be 'vtc' or 'lena'")
        if self.distribution not in ["poisson", "gamma"]:
            raise ValueError("Distribution must be 'poisson' or 'gamma'")

        self._load_hyperparameters()

    def _load_hyperparameters(self):
        """Load hyperparameters from the data files."""
        try:
            data_path = files(
                "diarization_simulation",
            ) / "data" / f"{self.algorithm}.npz"
            samples_location = str(data_path)
        except ImportError:
            # Fallback to relative path if package not found
            samples_location = f"data/{self.algorithm}.npz"

        if not os.path.exists(samples_location):
            raise FileNotFoundError(
                f"Hyperparameter file not found: {samples_location}",
            )

        self.samples = np.load(samples_location)
        self.n_available_samples = self.samples["mus"].shape[0]

        # Extract hyperparameters
        self.mu_ = self.samples["mus"]
        self.alpha_ = self.samples["alphas"]
        self.tau_ = self.samples["tau"]

        # Calculate means
        self.mu_mean = self.mu_.mean(axis=0)
        self.alpha_mean = self.alpha_.mean(axis=0)
        self.tau_mean = self.tau_.mean(axis=0)

    def simulate(
            self,
            truth_data: Union[str, pd.DataFrame],
            n_samples: int = 1000,
            hyperprior_mode: str = "sample",
            random_seed: Optional[int] = None,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate diarization detections.

        Args:
            truth_data: Path to CSV file or DataFrame containing true vocalization counts
            n_samples: Number of samples to generate per observation
            hyperprior_mode: How to handle hyperpriors:
                - "sample": Each sample gets its own hyperpriors (default)
                - "average": Use mean hyperprior values
                - "unique": Use same hyperpriors for all samples
            random_seed: Random seed for reproducibility
            verbose: Whether to show progress bar

        Returns:
            DataFrame with simulated detections
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Load truth data
        if isinstance(truth_data, str):
            recs = pd.read_csv(truth_data)
        else:
            recs = truth_data.copy()

        # Validate required columns
        if not set(self.speakers).issubset(recs.columns):
            raise ValueError(
                f"Truth data must contain all speakers: {self.speakers}",
            )

        n_recs = len(recs)
        true_vocs = np.stack(recs[self.speakers].values)
        observation = recs[
            "observation"].values if "observation" in recs.columns else np.arange(
            n_recs,
        )

        # Initialize detection arrays
        detected = {speaker: [] for speaker in self.speakers}
        detected["observation"] = []
        detected["sample"] = []

        # Set up hyperprior sampling
        drawn = np.random.choice(np.arange(self.n_available_samples), n_samples)
        distribution_type = 0 if self.distribution == "poisson" else 1

        # Warning messages
        if hyperprior_mode == "sample" and verbose:
            print(
                "\033[94mINFO\033[0m: Each sample will have its own hyperpriors. "
                "This captures uncertainty about algorithm behavior.",
            )
        elif hyperprior_mode == "unique" and verbose:
            print(
                "\033[93mWARNING\033[0m: Using unique hyperpriors for all samples. "
                "Consider generating multiple datasets with different seeds.",
            )

        # Simulation loop
        iterator = tqdm(
            enumerate(drawn), total=len(drawn), desc="Simulating",
        ) if verbose else enumerate(drawn)

        for n, s in iterator:
            # Select hyperpriors based on mode
            if hyperprior_mode == "unique":
                draw = drawn[0]
            elif hyperprior_mode == "sample":
                draw = s
            else:  # average
                draw = None

            if hyperprior_mode == "average":
                current_mu = self.mu_mean
                current_alpha = self.alpha_mean
                current_tau = self.tau_mean
            else:
                current_mu = self.mu_[draw]
                current_alpha = self.alpha_[draw]
                current_tau = self.tau_[draw]

            # Call the numba function
            sample_detections = simulate_sample(
                true_vocs,
                current_alpha,
                current_mu,
                current_tau,
                n_recs,
                len(self.speakers),
                distribution_type,
            )

            # Store results
            for k in range(n_recs):
                for j, speaker in enumerate(self.speakers):
                    detected[speaker].append(sample_detections[k, j])

                detected["observation"].append(observation[k])
                detected["sample"].append(n)

        # Create and return DataFrame
        detected_df = pd.DataFrame(detected)
        detected_df = detected_df.sort_values(
            ["sample", "observation"],
        ).reset_index(drop=True)

        return detected_df

    def save_results(
            self,
            results: pd.DataFrame,
            output_path: str,
            format: str = "csv",
    ):
        """
        Save simulation results to file.

        Args:
            results: DataFrame with simulation results
            output_path: Output file path
            format: Output format ("csv", "parquet", "npz")
        """
        output_path = Path(output_path)
        base_path = output_path.with_suffix('')

        if format == 'csv':
            output_file = base_path.with_suffix('.csv')
            results.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")

        elif format == 'parquet':
            output_file = base_path.with_suffix('.parquet')
            results.to_parquet(output_file, index=False)
            print(f"Results saved to: {output_file}")

        elif format == 'npz':
            output_file = base_path.with_suffix('.npz')

            # Convert DataFrame to numpy arrays for npz format
            data_dict = {}
            for col in results.columns:
                data_dict[col] = results[col].values

            np.savez_compressed(output_file, **data_dict)
            print(f"Results saved to: {output_file}")

        else:
            raise ValueError("Format must be 'csv', 'parquet', or 'npz'")


def simulate_diarization(
        truth_data: Union[str, pd.DataFrame],
        algorithm: str = "vtc",
        distribution: str = "poisson",
        n_samples: int = 1000,
        hyperprior_mode: str = "sample",
        random_seed: Optional[int] = None,
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to simulate diarization detections.

    This is a simplified interface to the DiarizationSimulator class.

    Args:
        truth_data: Path to CSV file or DataFrame containing true vocalization counts
        algorithm: Algorithm to simulate ("vtc" or "lena")
        distribution: Distribution type ("poisson" or "gamma")
        n_samples: Number of samples to generate per observation
        hyperprior_mode: How to handle hyperpriors ("sample", "average", "unique")
        random_seed: Random seed for reproducibility
        verbose: Whether to show progress bar

    Returns:
        DataFrame with simulated detections

    Example:
        >>> import pandas as pd
        >>> # Create sample truth data
        >>> truth_df = pd.DataFrame({
        ...     'observation': [1, 2, 3],
        ...     'CHI': [10, 15, 8],
        ...     'OCH': [5, 3, 12],
        ...     'FEM': [20, 18, 25],
        ...     'MAL': [12, 10, 15]
        ... })
        >>>
        >>> # Simulate detections
        >>> results = simulate_diarization(
        ...     truth_data=truth_df,
        ...     algorithm="vtc",
        ...     n_samples=100,
        ...     random_seed=42
        ... )
        >>> print(results.head())
    """
    simulator = DiarizationSimulator(
        algorithm=algorithm, distribution=distribution,
    )
    return simulator.simulate(
        truth_data=truth_data,
        n_samples=n_samples,
        hyperprior_mode=hyperprior_mode,
        random_seed=random_seed,
        verbose=verbose,
    )

def main():
    """Command-line interface (maintains backward compatibility)."""
    parser = argparse.ArgumentParser(
        description="Simulate diarization detections",
    )
    parser.add_argument(
        "--truth",
        required=True,
        help="Path to the synthetic truth dataset (in csv format)",
    )
    parser.add_argument(
        "--output", required=True, help="Location of the output file",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet", "npz"],
        default="csv",
        help="Output file format (default: csv)",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of samples per observation",
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
        "--algo", choices=["vtc", "lena"], required=True,
        help="Algorithm to simulate",
    )
    parser.add_argument(
        "--distribution",
        choices=["poisson", "gamma"],
        default="poisson",
        help="Distribution for vocalization counts",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Determine hyperprior mode
    if args.average_hyperpriors:
        hyperprior_mode = "average"
    elif args.unique_hyperpriors:
        hyperprior_mode = "unique"
    else:
        hyperprior_mode = "sample"

    # Create simulator and run simulation
    simulator = DiarizationSimulator(
        algorithm=args.algo,
        distribution=args.distribution,
    )

    results = simulator.simulate(
        truth_data=args.truth,
        n_samples=args.samples,
        hyperprior_mode=hyperprior_mode,
        random_seed=args.seed,
        verbose=True,
    )

    # Save results
    simulator.save_results(results, args.output, args.output_format)


if __name__ == "__main__":
    main()

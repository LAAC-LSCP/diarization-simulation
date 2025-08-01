import numpy as np
import pandas as pd
from scipy.stats import gamma, poisson

try:
    from cmdstanpy import CmdStanModel
    from ChildProject.projects import ChildProject
    from ChildProject.annotations import AnnotationManager
except ImportError as e:
    raise ImportError(
        f"Truth analysis functionality requires additional dependencies. "
        f"Install with: pip install cmdstanpy ChildProject\n"
        f"Original error: {e}",
    ) from e

import tempfile
import argparse


def generate_ground_truth(
        corpus: str, annotator: str, recordings_path: list = None,
        n_samples: int = 1000, mode: bool = False,
        show_distribution: bool = False,
):
    """
    :param corpus: path to the childproject corpus
    :param annotator: annotation set containing the manual annotations
    :param recordings: whitelist of recordings (optional)
    :param n_samples: number of samples to generate
    :param mode: if True, use mode point estimates instead of full posterior sampling
    :return: pd.DataFrame with simulated ground truth annotations and four columns
    """

    speakers = ["CHI", "OCH", "FEM", "MAL"]
    C = len(speakers)

    project = ChildProject(corpus)
    project.read()

    if recordings_path is None:
        recordings = project.recordings
    else:
        recordings = pd.read_csv(recordings_path)
        missing_columns = {
                              "recording_filename", "duration",
                          } - set(recordings.columns)
        if missing_columns:
            raise Exception(
                f"The following columns are missing from {recordings_path}: {missing_columns}",
            )

    recordings["duration"] /= 1000

    am = AnnotationManager(project)
    am.read()
    annotations = am.annotations[am.annotations["set"] == annotator]
    annotations["annotated_duration"] = (annotations["range_offset"] -
                                         annotations["range_onset"]) / 1000
    annotated_duration = annotations.groupby("recording_filename")[
        "annotated_duration"].sum().to_dict()
    segments = am.get_segments(annotations)
    segments = segments[segments["speaker_type"].isin(speakers)]

    annotated_data = []

    for recording_filename, _ in segments.groupby(
            "recording_filename",
    ):
        data = {
            speaker: 0
            for speaker in speakers
        }
        data["recording_filename"] = recording_filename
        data["annotated_duration"] = annotated_duration[recording_filename]

        for speaker, vocalizations in _.groupby("speaker_type"):
            data[speaker] = len(vocalizations)

        annotated_data.append(data.copy())

    annotated_data = pd.DataFrame(annotated_data)

    N = len(annotated_data)
    K = len(recordings)
    recording_filenames = recordings["recording_filename"].values

    # Prepare data for Stan
    data = {
        'N': N,
        'K': K,
        'C': C,
        'vocs': np.stack(annotated_data[speakers].values),
        'annotated_duration': annotated_data["annotated_duration"],
        'full_duration': recordings["duration"],
    }

    # Build model
    print("Building Stan model...")
    stan_code = """
    data {
      int<lower=0> N; // amount of manually annotated recordings
      int<lower=0> K; // amount of recordings in dataset
      int<lower=0> C; // amount of speakers
      array[N, C] int<lower=0> vocs;
      vector[N] annotated_duration;
      vector[K] full_duration;
    }

    parameters {
      array[N, C] real<lower=0> rates;
      vector<lower=0>[C] shape;
      vector<lower=0>[C] rate;
    }

    model {
      for (i in 1:N) {
        for (c in 1:C) {
          vocs[i, c] ~ poisson(rates[i, c] * annotated_duration[i]);
          rates[i, c] ~ gamma(shape[c], shape[c]/rate[c]);
        }
      }

      for (c in 1:C) {
        shape[c] ~ gamma(4.0, 4.0);
        rate[c] ~ exponential(1.0/(250.0/3600.0)); // assumes 250 vocs/hour
      }
    }

    generated quantities {
        array[K,C] int samples;
        for (k in 1:K) {
            for (c in 1:C) {
                samples[k,c] = poisson_rng(gamma_rng(shape[c], shape[c]/rate[c])*full_duration[k]);
            }
        }
    }
    """
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w', delete=False, suffix=".stan",
    )
    with tmp_file as tmp:
        tmp.write(stan_code)

    model = CmdStanModel(stan_file=tmp_file.name)

    # Full posterior sampling
    fit = model.sample(
        data=data, chains=1, iter_warmup=1000,
        iter_sampling=np.maximum(n_samples, 10000),
    )
    variables = fit.stan_variables()
    sampler = fit.method_variables()
    posterior_mode = sampler["lp__"].argmax()
    samples = variables["samples"]

    if show_distribution:
        from matplotlib import pyplot as plt
        PER_HOUR = 3600
        y = np.zeros((C, 100, samples.shape[0]))
        upper_bound = np.zeros((C, samples.shape[0]))

        fig, ax = plt.subplots()
        for c, speaker in enumerate(speakers):
            for s in range(samples.shape[0]):
                upper_bound[c, s] = gamma.ppf(
                    q=0.99,
                    a=variables["shape"][s, c],
                    scale=variables["rate"][s, c] / variables["shape"][s, c],
                ) * PER_HOUR

        x_max = np.max(np.mean(upper_bound, axis=1))
        ax.set_xlim(0, x_max)
        x = np.linspace(0, x_max, y.shape[1])

        for c, speaker in enumerate(speakers):
            for s in range(samples.shape[0]):
                y[c, :, s] = gamma.pdf(
                    x=x / PER_HOUR,
                    a=variables["shape"][s, c],
                    scale=variables["rate"][s, c] / variables["shape"][
                        s, c],
                ) / PER_HOUR

            ax.fill_between(
                x, np.quantile(y[c], q=0.05 / 2, axis=1),
                np.quantile(y[c], q=1 - 0.05 / 2, axis=1),
                alpha=0.25,
            )
            ax.plot(x, np.mean(y[c], axis=1), label=speaker)

        ax.set_xlabel("Vocalizations per hour")
        ax.set_ylabel("Density")

        plt.legend()
        plt.show()

    df = []
    if mode:
        for i in range(n_samples):
            for k in range(K):
                data_row = {}
                for c, speaker in enumerate(speakers):
                    # Sample rate from gamma distribution with mode parameters
                    rate = gamma.rvs(
                        a=variables["shape"][posterior_mode, c],
                        scale=variables["rate"][posterior_mode, c] /
                              variables["shape"][
                                  posterior_mode, c],
                    )
                    # Sample vocalizations from Poisson with this rate
                    data_row[speaker] = poisson.rvs(
                        rate * recordings["duration"].iloc[k],
                    )

                data_row["recording_filename"] = recording_filenames[k]
                data_row["observation"] = f"{recording_filenames[k]},{i}"
                df.append(data_row.copy())
    else:
        keep = np.random.choice(np.arange(samples.shape[0]), size=n_samples)

        for i, s in enumerate(keep):
            for k in range(K):
                data_row = {
                    speakers[c]: samples[s, k, c]
                    for c in range(C)
                }
                data_row["recording_filename"] = recording_filenames[k]
                data_row["observation"] = f"{recording_filenames[k]},{i}"
                df.append(data_row.copy())

    df = pd.DataFrame(df)
    return df


def main():
    """Command-line interface (maintains backward compatibility)."""
    parser = argparse.ArgumentParser(
        description="Simulate ground-truth vocalization counts for each recording of a corpus given sparse manual annotations",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to the input dataset",
    )
    parser.add_argument(
        "--annotator",
        required=True,
        help="Annotations to infer ground truth distribution from",
    )
    parser.add_argument(
        "--output", required=True, help="Location of the output file",
    )
    parser.add_argument(
        "--recordings", default=None,
        help="Path to list of recordings (CSV)",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of samples",
    )
    parser.add_argument(
        "--mode", action="store_true",
        default=False,
        help="Sample from the mode of the posterior distribution of the hyperparameters.",
    )
    parser.add_argument(
        "--show-distribution", action="store_true",
    )

    args = parser.parse_args()

    df = generate_ground_truth(
        args.corpus, args.annotator, args.recordings,
        n_samples=args.samples, mode=args.mode,
        show_distribution=args.show_distribution,
    )

    df.to_csv(args.output)

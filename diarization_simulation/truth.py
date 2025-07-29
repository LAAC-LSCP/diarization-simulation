import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy.stats import gamma, poisson
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import tempfile
import argparse


def generate_ground_truth(
        corpus: str, annotator: str, recordings: list = None,
        n_samples: int = 1000,
):
    """
    :param corpus: path to the childproject corpus
    :param annotator: annotation set containing the manual annotations
    :param recordings: whitelist of recordings (optional)
    :return: pd.DataFrame with simulated ground truth annotations and four columns
    """

    speakers = ["CHI", "OCH", "FEM", "MAL"]
    C = len(speakers)

    project = ChildProject(corpus)
    project.read()

    if recordings is None:
        recordings = project.recordings
    else:
        recordings = project.recordings[
            project.recordings["recording_filename"].isin(recordings)]

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
    print(annotated_data)

    N = len(annotated_data)
    K = len(recordings)

    # Prepare data for Stan
    data = {
        'N': N,
        'K': K,
        'C': C,
        'vocs': np.stack(annotated_data[speakers].values),
        'annotated_duration': annotated_data["annotated_duration"],
        'full_duration': recordings["duration"],
    }

    # Build and fit model
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
        shape[c] ~ exponential(1);
        rate[c] ~ exponential(4);
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
    fit = model.sample(
        data=data, chains=1, iter_warmup=1000,
        iter_sampling=np.maximum(n_samples, 1000),
    )
    variables = fit.stan_variables()

    recording_filenames = recordings["recording_filename"].values
    samples = variables["samples"]

    df = []
    keep = np.random.choice(np.arange(samples.shape[0]), size=n_samples)

    for i, s in enumerate(keep):
        for k in range(K):
            data = {
                speakers[c]: samples[s, k, c]
                for c in range(C)
            }
            data["observation"] = recording_filenames[k]
            data["sample"] = i
            df.append(data.copy())

    df = pd.DataFrame(df)
    return df


def main():
    """Command-line interface (maintains backward compatibility)."""
    parser = argparse.ArgumentParser(
        description="Simulate vocalization counts",
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
        "--recordings", nargs="+", default=None,
        help="Optional whitelist of recordings",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of samples",
    )

    args = parser.parse_args()

    df = generate_ground_truth(
        args.corpus, args.annotator, args.recordings, n_samples=args.samples,
    )

    df.to_csv(args.output)

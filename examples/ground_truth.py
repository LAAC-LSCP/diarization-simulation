import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from scipy.stats import gamma, poisson
from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

def generate_ground_truth(
        corpus: str, annotator: str, recordings: list = None,
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
    model = CmdStanModel(stan_file="examples/ground_truth.stan")

    print("Running MLE optimization...")
    fit = model.sample(data=data)
    variables = fit.stan_variables()

    recording_filenames = recordings["recording_filename"].values
    samples = variables["samples"]

    print(samples)

    df = []
    for s in range(samples.shape[0]):
        for k in range(K):
            data = {
                speakers[c]: samples[s, k, c]
                for c in range(C)
            }
            data["observation"] = recording_filenames[k]
            data["sample"] = s
            df.append(data.copy())

    df = pd.DataFrame(df)
    return df

df = generate_ground_truth(
    "../data/tsimane2017", "textgrid/mm",
)

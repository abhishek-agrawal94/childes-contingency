from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
import pandas as pd
import glob
import argparse

from utils import (PROJECT_ROOT_DIR)

DEFAULT_FILE_REGEX = PROJECT_ROOT_DIR + "/data/raw_annotations/*.csv"


def compute_agreement_over_dir(file_regex: str, speaker_code: str, metric: str) -> float:
    csv_files = glob.glob(file_regex)
    df_concat = pd.concat([pd.read_csv(f, delimiter=';') for f in csv_files], ignore_index=True)
    df_concat = df_concat[~df_concat['annotation'].isna()]
    if speaker_code == 'PAR':
        mask = df_concat.speaker_code == 'MOT'
        col_name = 'speaker_code'
        df_concat.loc[mask, col_name] = 'PAR'

        mask = df_concat.speaker_code == 'FAT'
        col_name = 'speaker_code'
        df_concat.loc[mask, col_name] = 'PAR'
    df_concat = df_concat[df_concat['speaker_code'] == speaker_code]
    anno1_values = df_concat['annotation']
    anno2_values = df_concat['annotation_mitja']
    if metric == "mcc":
        result = matthews_corrcoef(anno1_values, anno2_values)
    elif metric == "f1":
        result = f1_score(anno1_values, anno2_values, average='weighted')
    else:
        result = cohen_kappa_score(anno1_values, anno2_values, weights='quadratic')
    return result


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--file-regex",
        type=str,
        default=DEFAULT_FILE_REGEX,
        help="Regex for all files to compute inter-annotator agreement",
    )
    argparser.add_argument(
        "--speaker-code",
        type=str,
        default="CHI",
        help="Enter speaker code either 'CHI' or 'PAR'",
    )
    argparser.add_argument(
        "--metric",
        default="kappa",
        type=str,
        help="Choose the metric as either 'kappa', 'mcc' or 'f1'",
    )
    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.file_regex.endswith(".csv"):
        raise ValueError("File regex should end with .csv")
    result = compute_agreement_over_dir(args.file_regex, args.speaker_code, args.metric)
    print(result)
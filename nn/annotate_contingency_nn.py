import argparse
import glob
import os
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from pytorch_lightning import Trainer
from transformers import AutoTokenizer
import pandas as pd

from nn.data import load_annotated_childes_data_with_context, \
    CHILDESContingencyDataModule, prepare_new_england_data
from nn.fine_tuning_nn import CHILDESContingencyModel
from utils import PROJECT_ROOT_DIR

ANNOTATION_ANNOTATED_FILES_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "all_childes")
ANNOTATION_ZERO_CONTEXT = os.path.expanduser(os.path.join(PROJECT_ROOT_DIR, "data", "agreement"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Needs to match the number of utterances within a file to be annotated!
BATCH_SIZE = 200


def annotate(args):
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    context_length = args.context_length
    sep_token = tokenizer.sep_token
    data = load_annotated_childes_data_with_context(args.data_dir, context_length=context_length, sep_token=sep_token,
                                                    preserve_age_column=True, model_type=args.model_type)
    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset_dict = DatasetDict()
    dataset_dict["pred"] = dataset
    dm = CHILDESContingencyDataModule(val_split_proportion=0,
                                  num_cv_folds=0,
                                  model_name_or_path=args.model,
                                  eval_batch_size=BATCH_SIZE,
                                  train_batch_size=BATCH_SIZE,
                                  tokenizer=tokenizer,
                                  context_length=context_length,
                                  num_workers=args.num_workers,
                                  add_eos_tokens=False,
                                  train_data_size=1,
                                  ds_dict=dataset_dict)

    checkpoints = glob.glob(args.model+"/checkpoints/epoch*.ckpt")
    print(f"Model checkpoints: {checkpoints}")

    #data_annotated = load_annotated_childes_data(args.data_dir)
    data_annotated = prepare_new_england_data(args.data_dir, context_length, args.model_type)
    data_annotated = pd.DataFrame.from_records(data_annotated)

    for i, checkpoint in enumerate(checkpoints):
        print(f"\n\nAnnotating with model checkpoint #{i}")
        model = CHILDESContingencyModel.load_from_checkpoint(checkpoint, predict_data_dir=args.data_dir, model_id=i)
        model.eval()

        trainer = Trainer(devices=1 if torch.cuda.is_available() else None, accelerator="auto")
        predictions = trainer.predict(model, datamodule=dm)
        predictions = [x for preds in predictions for x in preds]
        print(f'Preds length: {len(predictions)}')
        data_annotated[f"is_coherent_{i}"] = predictions


    def majority_vote(row):
        votes = [row[f"is_coherent_{i}"] for i in range(len(checkpoints))]
        maj_vote = np.bincount(votes).argmax()
        maj_vote = maj_vote - 1
        return maj_vote

    data_annotated["coherence"] = data_annotated.apply(majority_vote, axis=1)

    # Append training data
    #data_train = load_childes_data(DATA_PATH_CHILDES_ANNOTATED)
    #data_all = pd.concat([data_train, data_annotated])

    data_annotated.to_csv(os.path.join(args.data_dir, "majority_vote_0_context_child.csv"))


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-dir",
        type=str,
        default=ANNOTATION_ZERO_CONTEXT,
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="path to model checkpoint"
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    argparser.add_argument(
        "--model-type",
        type=str,
        default='child',
        help="parent or child model"
    )
    argparser.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Number of preceding utterances to include as conversational context"
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    annotate(args)
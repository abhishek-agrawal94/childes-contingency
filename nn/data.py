import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import glob
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from nn.tokenizer import (TEXT_FIELD, LABEL_FIELD, TOKEN_SPEAKER_CHILD, TRANSCRIPT_FIELD,
                                                 TOKEN_SPEAKER_CAREGIVER)
from utils import SPEAKER_CODE_CHILD, SPEAKER_CODES_CAREGIVER


DATA_SPLIT_RANDOM_STATE = 8

DATA_PATH_CHILDES_ANNOTATED = os.path.expanduser(os.path.join("/gpfswork/rech/eqb/uez75lm", "data", "coherence_parent"))
DATA_PATH_NEW_ENGLAND_ANNOTATED = os.path.expanduser(os.path.join("/gpfswork/rech/eqb/uez75lm", "data", "agreement"))


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def speaker_code_to_speaker_token(code):
    if code == SPEAKER_CODE_CHILD:
        return TOKEN_SPEAKER_CHILD
    if code in SPEAKER_CODES_CAREGIVER:
        return TOKEN_SPEAKER_CAREGIVER
    raise RuntimeError("Unknown speaker code: ", code)

def create_cv_folds(data, num_folds):
    transcript_files_sizes = data.transcript_file.value_counts()

    test_sets = [pd.DataFrame() for _ in range(num_folds)]

    while len(transcript_files_sizes) > 0:
        smallest_fold = np.argmin([len(s) for s in test_sets])
        largest_transcript = transcript_files_sizes.index[0]

        # Add the largest transcript to the smallest fold
        test_sets[smallest_fold] = pd.concat([test_sets[smallest_fold], data[data.transcript_file == largest_transcript]])
        del transcript_files_sizes[largest_transcript]

    train_sets = [data[~data.index.isin(data_test.index)].copy() for data_test in test_sets]

    #print(f"Test set sizes: {[len(s) for s in test_sets]}")
    return train_sets, test_sets

def train_val_split(data, val_split_size, random_seed=DATA_SPLIT_RANDOM_STATE):
    # Make sure that test and train split do not contain data from the same transcripts
    if isinstance(val_split_size, float):
        train_data_size = int(len(data) * (1 - val_split_size))
    else:
        train_data_size = len(data) - val_split_size
    transcript_files = data.transcript_file.unique()
    random.seed(random_seed)
    random.shuffle(transcript_files)
    transcript_files = iter(transcript_files)
    data_train = pd.DataFrame()
    # Append transcripts until we have the approximate train data size.
    while len(data_train) < train_data_size:
        data_train = pd.concat([data_train, data[data.transcript_file == next(transcript_files)]])

    data_val = data[~data.index.isin(data_train.index)].copy()

    assert (len(set(data_train.index) & set(data_val.index)) == 0)
    return data_train, data_val

def get_train_test_split(data, test_split_size, random_seed=DATA_SPLIT_RANDOM_STATE):
    # Make sure that test and train split do not contain data from the same transcripts
    # if isinstance(test_split_size, float):
    #     train_data_size = int(len(data) * (1 - test_split_size))
    # else:
    #     train_data_size = len(data) - test_split_size
    # transcript_files = data.transcript_file.unique()
    random.seed(random_seed)
    # random.shuffle(transcript_files)
    # transcript_files = iter(transcript_files)
    data_train, data_test = train_test_split(data, test_size=test_split_size)
    # Append transcripts until we have the approximate train data size.
    # while len(data_train) < train_data_size:
    #     data_train = pd.concat([data_train, data[data.transcript_file == next(transcript_files)]])

    # data_test = data[~data.index.isin(data_train.index)].copy()

    assert (len(set(data_train.index) & set(data_test.index)) == 0)

    return data_train, data_test


def load_annotated_childes_data(path):
    transcripts = []
    for f in Path(path).glob("all_childes_20_and_above_all_parent*.json"):
        if os.path.isfile(f):
            transcripts.append(pd.read_json(f))

    transcripts = pd.concat(transcripts, ignore_index=True)
    # transcripts["speaker_code"] = transcripts.speaker_code.apply(speaker_code_to_speaker_token)
    # transcripts["sentence"] = transcripts.apply(lambda row: row.speaker_code + row.transcript_clean,
    #                                                 axis=1).values
    print("Transcripts size:", len(transcripts))
    return transcripts

def load_annotated_childes_data_with_context(path=DATA_PATH_NEW_ENGLAND_ANNOTATED, context_length=0, sep_token=None,
                                             preserve_age_column=False, model_type="child"):
    transcripts = prepare_new_england_data(path, context_length, model_type)
    transcripts = pd.DataFrame.from_records(transcripts)
    #transcripts = load_annotated_childes_data(path)
    data = []
    for i, row in transcripts.iterrows():
        sentence = row.context + sep_token + row.turn
        datapoint = {
            TEXT_FIELD: sentence,
            TRANSCRIPT_FIELD: row[TRANSCRIPT_FIELD],
            #"turn_utterance_id": row.turn_utterance_id
        }
        if preserve_age_column:
            datapoint["age"] = row.age
        data.append(datapoint)

    data = pd.DataFrame.from_records(data)


    print("Dataset size: ", len(data))
    return data

def prepare_new_england_data(path=DATA_PATH_NEW_ENGLAND_ANNOTATED, context_length=0, model_type="child"):
    csv_files = glob.glob(path + '/*')
    files = []
    for f in csv_files:
        df = pd.read_csv(f, delimiter=';')
        age = int(f.split('/')[-1].split('.')[0].split('_')[-1])
        length = df.shape[0]
        df['age'] = [age] * length
        files.append(df)
    df_concat = pd.concat(files, ignore_index=True)
    utterances, child_utts, parent_utts = [], [], []
    for index, row in df_concat.iterrows():
        temp_dict = {}
        temp_dict['speaker_code'] = row['speaker_code']
        temp_dict['text'] = row['transcript_clean']
        temp_dict['annotation'] = row['coherence_agreement']
        temp_dict['speech_act_contingency'] = row['speech_act_contingency']
        temp_dict['speech_act'] = row['speech_act']
        temp_dict['age'] = row['age']
        temp_dict['transcript_id'] = row['transcript_id']
        utterances.append(temp_dict)
        if row['speaker_code'] == 'CHI' and pd.notna(row['coherence_agreement']):
            child_utts.append(temp_dict)
        elif (row['speaker_code'] == 'MOT' or row['speaker_code'] == 'FAT') and pd.notna(row['coherence_agreement']):
            parent_utts.append(temp_dict)

    if model_type == "child":
        utts = child_utts
    else:
        utts = parent_utts

    candidate_turn_data = []
    for utt in utts:
        index = utterances.index(utt)
        if context_length >= index:
            turn_list = utterances[:index]
        else:
            turn_list = utterances[index - context_length:index]

        context = ""
        if context_length > 0:
            for item in turn_list:
                context += speaker_code_to_speaker_token(item['speaker_code']) + " " + item['text'] + " "

        obj = {
                "context": context,
                "transcript_file": utt['transcript_id'],
                "turn": speaker_code_to_speaker_token(utt['speaker_code']) + " " + utt['text'],
                "annotation": utt['annotation'],
                "age": utt['age']
                }
        candidate_turn_data.append(obj)

    print(f'Total turn switches: {len(candidate_turn_data)}')

    return candidate_turn_data

def load_annotated_childes_datasplits(context_length=0, num_cv_folds=10, sep_token=None, age=None, model_type="child"):
    transcripts = prepare_new_england_data(DATA_PATH_NEW_ENGLAND_ANNOTATED, context_length, model_type)
    transcripts = pd.DataFrame.from_records(transcripts)
    #transcripts = load_annotated_childes_data(DATA_PATH_CHILDES_ANNOTATED)
    if age:
        transcripts = transcripts[transcripts['age'] == age]
    data = []
    print(f'Label field:{LABEL_FIELD}')
    for i, row in transcripts[~transcripts[LABEL_FIELD].isna()].iterrows():
        sentence = row.context
        if sep_token:
            sentence += sep_token
        sentence += row.turn
        # for j in range(1, context_length+1):
        #     if i-j in transcripts.index:
        #         context_sentence = transcripts.loc[i-j].sentence
        #         sentence = context_sentence + sentence
        data.append({
            TEXT_FIELD: sentence,
            LABEL_FIELD: row[LABEL_FIELD],
            TRANSCRIPT_FIELD: row[TRANSCRIPT_FIELD]
        })
    data = pd.DataFrame.from_records(data)

    # Transform -1, 0, 1 to 0, 1, 2 so that they can be of dtype long
    data[LABEL_FIELD] = (data[LABEL_FIELD] + 1).astype("int64")

    print("Dataset size: ", len(data))

    return create_cv_folds(data, num_cv_folds)


LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        LABEL_FIELD,
    ]

def create_dataset_dicts(num_cv_folds, val_split_proportion, context_length, random_seed=DATA_SPLIT_RANDOM_STATE, train_data_size=1.0,  create_val_split=False, sep_token=None, age=None, model_type="child"):
    dataset_dicts = [DatasetDict() for _ in range(num_cv_folds)]

    data_manual_annotations_train_splits, data_manual_annotations_test_splits = load_annotated_childes_datasplits(context_length, num_cv_folds, sep_token, age, model_type)
    if train_data_size < 1.0:
        data_manual_annotations_train_splits = [d.sample(round(len(d) * train_data_size), random_state=DATA_SPLIT_RANDOM_STATE) for d in data_manual_annotations_train_splits]
    for fold in range(num_cv_folds):
        if create_val_split:
            data_manual_annotations_train_splits[fold], data_manual_annotations_val_split = train_val_split(data_manual_annotations_train_splits[fold], val_split_proportion, random_seed)
            ds_val = Dataset.from_pandas(data_manual_annotations_val_split, preserve_index=False)
            dataset_dicts[fold]["validation"] = ds_val

        ds_train = Dataset.from_pandas(data_manual_annotations_train_splits[fold], preserve_index=False)
        dataset_dicts[fold]["train"] = ds_train

        ds_test = Dataset.from_pandas(data_manual_annotations_test_splits[fold], preserve_index=False)
        dataset_dicts[fold]["test"] = ds_test

    return dataset_dicts

class CHILDESContingencyDataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str,
            train_batch_size: int,
            eval_batch_size: int,
            ds_dict: DatasetDict,
            tokenizer,
            max_seq_length: int = 128,
            num_cv_folds = 10,
            val_split_proportion: float = 0.2,
            context_length: int = 1,
            random_seed=1,
            num_workers=8,
            add_eos_tokens=False,
            train_data_size=1.0,
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_cv_folds= num_cv_folds
        self.val_split_proportion = val_split_proportion
        self.context_length = context_length
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.train_data_size = train_data_size
        self.dataset = ds_dict

        self.num_labels = 3
        self.tokenizer = tokenizer
        self.add_eos_tokens = add_eos_tokens

    def setup(self, stage: str):
        for split in self.dataset.keys():
            columns = [c for c in self.dataset[split].column_names if c in LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=columns + [TEXT_FIELD])

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True,
                          collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size,
                          collate_fn=self.tokenize_batch, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, collate_fn=self.tokenize_batch,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset["pred"], batch_size=self.eval_batch_size,
                          collate_fn=self.tokenize_inference_batch, num_workers=self.num_workers)

    def tokenize_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_labels=True,
                        add_eos_token=self.add_eos_tokens)

    def tokenize_inference_batch(self, batch):
        return tokenize(batch, self.tokenizer, self.max_seq_length, add_eos_token=self.add_eos_tokens,
                        add_labels=False)

def tokenize(batch, tokenizer, max_seq_length, add_labels=False, add_eos_token=False):
    texts = [b[TEXT_FIELD] for b in batch]
    if add_eos_token:
        texts = [t + tokenizer.eos_token for t in texts]

    features = tokenizer.batch_encode_plus(
        texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt"
    )
    if add_labels:
        features.data[LABEL_FIELD] = torch.tensor([b[LABEL_FIELD] for b in batch])

    return features

def calc_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

    return class_weights
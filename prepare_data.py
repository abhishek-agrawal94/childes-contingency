import pandas as pd
import argparse
import glob
import json
from tqdm import tqdm
from utils import SPEAKER_CODES_CAREGIVER, PROJECT_ROOT_DIR

DEFAULT_FILE_PATH = PROJECT_ROOT_DIR + "/data/utterances_with_speech_acts_all_childes_no_new_england.csv"


def prepare_data_feature_extraction(file_path: str, target_speaker_code: str) -> (list, list):
    csv_files = glob.glob(file_path)
    files = []
    for f in csv_files:
        df = pd.read_csv(f, delimiter=';')
        age = int(f.split('/')[-1].split('.')[0].split('_')[-1])
        length = df.shape[0]
        df['age'] = [age] * length
        files.append(df)
    df_concat = pd.concat(files, ignore_index=True)

    utterances, speaker_utts = [], []

    for index, row in df_concat.iterrows():
        temp_dict = {}

        temp_dict['speaker_code'] = row['speaker_code']
        temp_dict['text'] = row['transcript_clean']
        # temp_dict['annotation'] = row['Agreement']
        temp_dict['annotation'] = row['coherence_agreement']
        temp_dict['speech_act_contingency'] = row['speech_act_contingency']
        temp_dict['speech_act'] = row['speech_act']
        temp_dict['age'] = row['age']
        temp_dict['transcript_id'] = row['transcript_id']

        utterances.append(temp_dict)

        row_speaker_code = row['speaker_code']
        if row['speaker_code'] == 'MOT' or row['speaker_code'] == 'FAT':
            row_speaker_code = 'PAR'


        # to sample the child/caregiver utterances without going through the entire list of child+adult
        if target_speaker_code == row_speaker_code and pd.notna(row['coherence_agreement']):
            speaker_utts.append(temp_dict)

    # print(len(utterances))
    return (utterances, speaker_utts)


# function for preparing data for fine-tuning for perplexity
def prepare_ppl_fine_tune_data(file_path: str, speaker_code: str, context_size: int) -> list:
    df = pd.read_csv(file_path)
    df = df[(df['is_speech_related'] == True) & (df['is_intelligible'] == True)]
    df = df[df['transcript_clean'].apply(lambda x: len(x.strip()) > 1)]
    df = df[df['speaker_code'] != 'INV']
    df = df[(df['age'] > 25) & (df['age'] <= 60)]
    df.drop(inplace=True,
            columns=['Unnamed: 0', 'transcript_raw', 'tokens', 'pos', 'gra', 'start_time', 'end_time', 'corpus',
                     'child_name', 'error', 'is_speech_related', 'is_intelligible'])

    data_dict = {}
    for index, row in tqdm(df.iterrows()):
        temp_dict = {}
        temp_dict['utterance_id'] = row['utterance_id']
        temp_dict['speaker_code'] = row['speaker_code']
        temp_dict['text'] = row['transcript_clean']

        # separate out all utterances belonging to a particular chat transcript as separate dict items
        if row['transcript_file'] in data_dict:

            data_dict[row['transcript_file']]['utterances'].append(temp_dict)
        else:
            data_dict[row['transcript_file']] = {
                "age": row['age'],
                "utterances": [temp_dict],
                "child_utts": [],
                "parent_utts": []
            }

        # to sample the child/caregiver utterances without going through the entire list of child+adult
        if row['speaker_code'] == 'CHI':
            data_dict[row['transcript_file']]['child_utts'].append(temp_dict)
        elif row['speaker_code'] in SPEAKER_CODES_CAREGIVER:
            data_dict[row['transcript_file']]['parent_utts'].append(temp_dict)

    #print(len(data_dict))

    # creating the actual turn data
    candidate_turn_data = []
    for transcript_file, trans_dict in tqdm(data_dict.items()):

        if speaker_code == 'CHI':
            speaker_id = 'child_utts'
        else:
            speaker_id = 'parent_utts'

        # if there are no intelligible child/parent utterances skip
        if not trans_dict[speaker_id]:
            continue

        # filter out instances for insertion point where the child/parent utterance is first or last in conversation
        # to have some context and next turn
        if trans_dict['utterances'].index(trans_dict[speaker_id][0]) == 0 and trans_dict['utterances'].index(
                trans_dict[speaker_id][-1]) == (len(trans_dict['utterances']) - 1):
            samples = trans_dict[speaker_id][1:-1]
        elif trans_dict['utterances'].index(trans_dict[speaker_id][0]) == 0:
            samples = trans_dict[speaker_id][1:]
        elif trans_dict['utterances'].index(trans_dict[speaker_id][-1]) == (len(trans_dict['utterances']) - 1):
            samples = trans_dict[speaker_id][:-1]
        else:
            samples = trans_dict[speaker_id]

        insertion_points = []
        for sample in samples:
            index = trans_dict['utterances'].index(sample)
            if trans_dict['utterances'][index - 1]['speaker_code'] != trans_dict['utterances'][index]['speaker_code']:
                insertion_points.append(sample)

        for insertion in insertion_points:
            # get the index of the insertion point for slice operations
            index = trans_dict['utterances'].index(insertion)
            if index >= context_size:
                # get turns for context
                turn_list = trans_dict['utterances'][index - context_size:index]

            else:
                turn_list = trans_dict['utterances'][:index]

            context = ""
            for item in turn_list:
                context += item['text'] + " "

            context += trans_dict['utterances'][index]['text']

            obj = {
                "text": context.strip()
            }
            candidate_turn_data.append(obj)

    #print(len(candidate_turn_data))
    return candidate_turn_data

# function for preparing files for manual annotation of contingency
def prepare_files_for_annotation(file_path: str, result_path: str):
    df = pd.read_csv(file_path)
    df = df[(df['is_speech_related'] == True) & (df['is_intelligible'] == True)]
    df = df[df['transcript_clean'].apply(lambda x: len(x.strip()) > 1)]
    df = df[df['speaker_code'] != 'INV']
    df.drop(inplace=True,
            columns=['Unnamed: 0', 'transcript_raw', 'tokens', 'pos', 'gra', 'start_time', 'end_time', 'age', 'corpus',
                     'child_name', 'error', 'is_speech_related', 'is_intelligible'])

    annotation, note, turn_switch, speaker_code, utterance_id, transcript, transcript_id = [], [], [], [], [], [], []
    count = 1
    tot_count = 0
    prev_turn = "INV"
    prev_trans = "14/01"
    prev_count = 1

    for index, row in df.iterrows():
        # 200 here is the batch size we used to annotate files at one go. Can be modified to any value
        if (count % 200 == 0) and (count != prev_count):
            temp_df = pd.DataFrame()
            temp_df['utterance_id'] = utterance_id
            temp_df['speaker_code'] = speaker_code
            temp_df['transcript_clean'] = transcript
            temp_df['annotation'] = annotation
            temp_df['turn_switch'] = turn_switch
            temp_df['transcript_id'] = transcript_id
            temp_df.to_csv(result_path + "/intelligible_utterances_new_england_" + str(tot_count) + ".csv")
            tot_count += 1
            prev_count = count

            annotation, note, turn_switch, speaker_code, utterance_id, transcript, transcript_id = [], [], [], [], [], [], []

        turn = row['speaker_code']
        transcript_num = row['transcript_file'].split("/")[1] + "/" + row['transcript_file'].split("/")[-1].split(".")[
            0]
        trans_id = row['transcript_file'].split("/")[1] + "-" + row['transcript_file'].split("/")[-1]
        if prev_trans != transcript_num:
            prev_turn = "INV"
            prev_trans = transcript_num
        transcript_id.append(trans_id)
        annotation.append('')
        speaker_code.append(turn)
        utterance_id.append(row['utterance_id'])
        transcript.append(row['transcript_clean'])
        if prev_turn != turn:
            if prev_turn == "INV":
                turn_switch.append('')
            else:
                turn_switch.append(1)
                count += 1
            prev_turn = turn
            # count += 1
        else:
            turn_switch.append('')

# function for preparing files for automatic annotation of contingency
def prepare_automatic_annotation_data(file_path: str, target_speaker_code: str, context_size: int, result_file: str):
    df = pd.read_csv(file_path, keep_default_na=False)
    df['is_speech_related'] = df['is_speech_related'].astype('bool')
    df['is_intelligible'] = df['is_intelligible'].astype('bool')
    df.fillna({'is_speech_related': False, 'is_intelligible': False})
    df = df[(df['is_speech_related'] == True) & (df['is_intelligible'] == True)]
    df = df[df['transcript_clean'].apply(lambda x: len(x.strip()) > 1)]
    df = df[df['speaker_code'] != 'INV']
    df.drop(inplace=True,
            columns=['Unnamed: 0', 'transcript_raw', 'tokens', 'pos', 'gra', 'start_time', 'end_time', 'corpus',
                     'child_name', 'error', 'is_speech_related', 'is_intelligible'])

    count = 0
    prev_turn = "INV"
    prev_trans = ""
    utterances, speaker_utts = [], []

    for index, row in tqdm(df.iterrows()):

        if row['age'] >= 20:
            turn = row['speaker_code']
            file_name = row['transcript_file']
            if prev_trans != file_name:
                prev_turn = "INV"
                prev_trans = file_name

            temp_dict = {}

            temp_dict['speaker_code'] = row['speaker_code']
            temp_dict['text'] = row['transcript_clean']
            temp_dict['speech_act_contingency'] = row['speech_act_contingency']
            temp_dict['speech_act'] = row['speech_act']
            temp_dict['transcript_file'] = row['transcript_file']
            temp_dict['age'] = row['age']
            temp_dict['utterance_id'] = row['utterance_id']

            utterances.append(temp_dict)
            # note.append('')
            if prev_turn != turn:
                if prev_turn != "INV":
                    row_speaker_code = row['speaker_code']
                    if row['speaker_code'] in SPEAKER_CODES_CAREGIVER:
                        row_speaker_code = 'PAR'

                    # to sample the child/caregiver utterances without going through the entire list of child+adult
                    if target_speaker_code == row_speaker_code:
                        speaker_utts.append(temp_dict)
                    count += 1
                prev_turn = turn

    #print(len(utterances))

    # creating the actual data for testing
    candidate_turn_data = []
    for utt in speaker_utts:
        index = utterances.index(utt)

        if context_size >= index:
            turn_list = utterances[:index]
        else:
            # get turns for context
            turn_list = utterances[index - context_size:index]

        context = ""
        concat_speech_acts = ""
        for item in turn_list:
            context += item['text'] + " "
            concat_speech_acts += item['speech_act']

        obj = {
            "context": context,
            "concat_speech_acts": concat_speech_acts,
            "turn": utt['text'],
            "transcript_file": utt['transcript_file'],
            "age": utt['age'],
            "turn_utterance_id": utt['utterance_id'],
            "speaker_code": utt['speaker_code'],
            "speech_act_contingency": utt['speech_act_contingency']
        }

        candidate_turn_data.append(obj)

    #print(len(candidate_turn_data))
    with open(result_file, "w") as fp:
        json.dump(candidate_turn_data, fp, indent=3)



def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--read-file-path",
        type=str,
        default=DEFAULT_FILE_PATH,
        help="File path for csv",
    )
    argparser.add_argument(
        "--write-directory-path",
        type=str,
        help="Directory path for writing data",
    )
    argparser.add_argument(
        "--speaker-code",
        type=str,
        default="CHI",
        help="Enter speaker code either 'CHI' or 'PAR'",
    )
    argparser.add_argument(
        "--context-size",
        default=5,
        type=int,
        help="Choose the context size in terms of turns",
    )
    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

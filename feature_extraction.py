import pandas as pd
import random
import json
import glob
import torch
import argparse
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from prepare_data import prepare_data_feature_extraction
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_all_features(file_path: str, speaker_code: str, context_size: int, ppl_model_id: str, results_path: str):
    spacy_nlp = spacy.load("en_core_web_trf")
    sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(ppl_model_id)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
    max_length = 512
    stride = 50
    utterances, speaker_utts = prepare_data_feature_extraction(file_path, speaker_code)

    candidate_turn_data = []
    for utt in tqdm(speaker_utts):
        index = utterances.index(utt)

        # sample number of context turns between 1-10
        # history_turns = random.randint(1, 10)

        if context_size >= index:
            turn_list = utterances[:index]
        else:
            # get turns for context
            turn_list = utterances[index - context_size:index]

        context = ""
        concat_speech_acts = ""
        concat_collapsed_speech_acts = ""
        for item in turn_list:
            context += item['text'] + " "
            concat_speech_acts += item['speech_act']


        # Extract noun, pronoun & noun phrase repetition counts between turn & context
        context_doc = spacy_nlp(context)
        context_noun_list, context_pron_list, context_noun_phrase = [], [], []
        for token in context_doc:
            if (token.pos_ == "PROPN") or (token.pos_ == "NOUN"):
                context_noun_list.append(token.lower_)
            elif token.pos_ == "PRON":
                context_pron_list.append(token.lower_)

        for np in context_doc.noun_chunks:
            context_noun_phrase.append(np.text.lower())

        turn_doc = spacy_nlp(utt['text'])
        turn_noun_list, turn_pron_list, turn_noun_phrase = [], [], []
        noun_repetitions, pron_repetitions, np_repetitions = 0, 0, 0
        for token in turn_doc:
            if (token.pos_ == "PROPN") or (token.pos_ == "NOUN"):
                turn_noun_list.append(token.lower_)
            elif token.pos_ == "PRON":
                turn_pron_list.append(token.lower_)

        for np in turn_doc.noun_chunks:
            turn_noun_phrase.append(np.text.lower())

        if turn_noun_list and context_noun_list:
            for noun in turn_noun_list:
                if noun in context_noun_list:
                    noun_repetitions += 1

        if turn_pron_list and context_pron_list:
            for pron in turn_pron_list:
                if pron in context_pron_list:
                    pron_repetitions += 1

        if turn_noun_phrase and context_noun_phrase:
            for np in turn_noun_phrase:
                if np in context_noun_phrase:
                    np_repetitions += 1

        # extract embeddings
        combined_representation = context + utt['text']
        combined_representation = sentence_transformer_model.encode(combined_representation)
        context_embedding = sentence_transformer_model.encode(context.strip(), convert_to_tensor=True)
        turn_embedding = sentence_transformer_model.encode(utt['text'].strip(), convert_to_tensor=True)

        # compute cosine similarity between context & turn embedding
        cos_sim = util.cos_sim(context_embedding, turn_embedding)

        # computing perplexity of turn given the context
        context_encoding = tokenizer(context, return_tensors="pt")
        turn_encoding = tokenizer(utt['text'], return_tensors="pt")
        encoding = torch.cat((context_encoding.input_ids, turn_encoding.input_ids), 1)
        seq_len = encoding.size(1)
        nlls = []
        token_count = 0
        prev_end_loc = context_encoding.input_ids.size(1)

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encoding[:, begin_loc:end_loc].to(device)
            input_ids = input_ids.type(torch.LongTensor)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids.to(device), labels=target_ids.to(device))
                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            token_count += trg_len
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / token_count)

        obj = {
            "context": context,
            "concat_speech_acts": concat_speech_acts + utt['speech_act'],
            "transcript_file": utt['transcript_id'],
            "turn": utt['text'],
            "annotation": utt['annotation'],
            "speech_act_contingency": utt['speech_act_contingency'],
            "age": utt['age'],
            "noun_repetitions": noun_repetitions,
            "pronoun_repetitions": pron_repetitions,
            "noun_phrase_repetitions": np_repetitions,
            "combined_representation": combined_representation.tolist(),
            "cosine_similarity": cos_sim.item(),
            "perplexity": ppl.detach().cpu().numpy().tolist()
        }

        candidate_turn_data.append(obj)

    #print(len(candidate_turn_data))

    with open(results_path, "w") as fp:
        json.dump(candidate_turn_data, fp, indent=3)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--read-file-path",
        type=str,
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

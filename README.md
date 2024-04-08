# CHILDES Contingency Annotations

## Python environment

A python environment can be created using the [environment.yml](environment/environment.yml) file.

```
conda env create --file environment/environment.yml
```

## Preprocess data

To preprocess all the CHILDES data, you will need to additionally install the [fork of the pylangacq repo](https://github.com/mitjanikolaus/pylangacq) by Mitja Nikolaus.

```
git clone git@github.com:mitjanikolaus/pylangacq.git
cd pylangacq
source activate cf
pip install .
```

### CHILDES corpora
All English CHILDES corpora need to be downloaded from the
[CHILDES database](https://childes.talkbank.org/).

To preprocess the data, once you've installed the [pylangacq](https://github.com/mitjanikolaus/pylangacq) library as
mentioned above, you can run:

```
python preprocess.py
```
This preprocesses all corpora that are conversational (have child AND caregiver transcripts), and are in English.

Afterwards, the utterances need to be annotated with speech acts. Use the method `crf_annotate` from the following
repository: [childes-speech-acts](https://github.com/mitjanikolaus/childes-speech-acts).
```
python crf_annotate.py --model checkpoint_full_train --data ~/data/communicative_feedback/utterances_annotated.csv --out ~/data/communicative_feedback/utterances_with_speech_acts.csv --use-pos --use-bi-grams --use-repetitions
```

Finally, annotate speech-relatedness and intelligibility (this is used to filter out non-speech-like utterances and
non-intelligible utterances before annotating contingency):
```
python annotate_speech_related_and_intelligible.py
```

### Contingency preprocessing

The [prepare_data.py](prepare_data.py) script contains all the functions for pre-processing the intelligible and speech-like data and
making it ready for contingency annotation. 

The manually annotated data is available in the [manually_annotated_data folder](data/manually_annotated_data).

## Training models for contingency annotation

### Feature-based models

First, run the [feature_extraction.py](feature_extraction.py) script to obtain all the relevant features for the context-turn
pairs.

```
python feature_extraction.py --read-file-path your_path_to_data --speaker-code 'CHI' --context-size 5
```

Then, run the [feature_based_classifier.py](feature_based_classifier.py) script to train the classifier.

```
python feature_based_classifier.py --data-path your_json_data_path --model "regression" --cv-folds 5 
```

### Language-model based approach

These models are fine-tuned on the task. Example for DeBERTa v3:

```
python nn/fine_tuning_nn.py --model microsoft/deberta-v3-large --context-length 5 --num-cv-folds 5 --model-type "child"
```
## Annotate data

To annotate data with the fine-tuned language models run the script [annotate_contingency_nn.py](nn/annotate_contingency_nn.py).

```
python nn/annotate_contingency_nn.py --model lightning_logs/version_123 --data-dir your_path_to_data --context-length 5 --model-type 'child'
```
import os


PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results_coherence.csv")
SPEAKER_CODES_CAREGIVER = [
    "MOT",
    "FAT",
    "DAD",
    "MOM",
    "GRA",
    "GRF",
    "GRM",
    "GMO",
    "GFA",
    "CAR",
]

SPEAKER_CODE_CHILD = "CHI"

FEATURE_MAP = {
    "baseline_random": ["random"],
    "baseline_majority": ["majority"],
    "ppl_default_gpt2": ["perplexity_default_gpt2"],
    "ppl_finetuned_gpt2": ["perplexity_finetuned_gpt2"],
    "SA": ["concat_speech_acts"],
    "NP_reps": ["np_repetitions"],
    "cos_sim": ["cosine_similarity"],
    "PPL+NP": ["perplexity_finetuned_gpt2", "np_repetitions"],
    "PPL+cos_sim": ["perplexity_finetuned_gpt2", "cosine_similarity"],
    "NP+cos_sim": ["np_repetitions", "cosine_similarity"],
    "PPL+NP+cos_sim": ["perplexity_finetuned_gpt2", "np_repetitions", "cosine_similarity"]
}
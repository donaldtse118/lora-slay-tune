# inbuilt
import os

# 3rd parties
# ml
from transformers import AutoTokenizer, AutoModel

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASET_NAME = 'gen_z_slang'
EVALUATE_DATASET_PATH = os.path.join("data", "fine_tuning", f"{DATASET_NAME}_evaluation.jsonl")
TRAINING_DATASET_PATH = os.path.join("data", "fine_tuning", f"{DATASET_NAME}_training.jsonl")
FINE_TUNED_MODEL_PATH = os.path.join("data", "model", f"{BASE_MODEL_NAME}-ft-{DATASET_NAME}")

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME)       
    lora_model = AutoModel.from_pretrained(FINE_TUNED_MODEL_PATH)
    base_model.eval()
    lora_model.eval()
    return base_model, lora_model, tokenizer
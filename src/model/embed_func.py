# inbuilt
import os
import logging

# 3rd parties
# ml
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# local imports
from model.mean_pooling import mean_pooling

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

_MODEL = None
_TOKENIZER = None

RESOURCE_PATH = "data"

_LOCAL_MODEL_PATH = os.path.join(RESOURCE_PATH, "model")

logger = logging.getLogger(__name__)
# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_model_and_tokenizer(model_name: str) -> tuple:
    """
    Retrieves the model and tokenizer for the specified model name.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: The loaded model and tokenizer.
    """

    global _MODEL, _TOKENIZER

    # Load your model here (consider using broadcast variables for efficiency)
    if _MODEL is None or _TOKENIZER is None:
        # Load the model and tokenizer only once
        logger.debug("Loading model and tokenizer...")

        model_ref = model_name
        local_model_path = os.path.join(_LOCAL_MODEL_PATH, model_name)
        if os.path.exists(local_model_path):
            model_ref = local_model_path
            logger.debug("Loading model and tokenizer from local directory...")

        _MODEL = AutoModel.from_pretrained(model_ref)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_ref)
        logger.debug("Model and tokenizer loaded.")

        if not os.path.exists(local_model_path):
            logger.debug("saving model and tokenizer to local directory...")
            save_model_and_tokenizer(_MODEL, _TOKENIZER, local_model_path)

    return _MODEL, _TOKENIZER


def save_model_and_tokenizer(model: AutoModel, tokenizer: AutoTokenizer, save_directory: str):

    # Save the model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)


def batch_generate_embedding(texts: pd.Series, model_name_or_path: str = MODEL_NAME) -> pd.Series:
    """
    Generates embeddings for a batch of input texts using a pre-trained model.

    Args:
        texts (pd.Series): A pandas Series containing the input texts for which embeddings are to be generated.

    Returns:
        pd.Series: A pandas Series containing the generated embeddings as lists of floats.
    """

    model, tokenizer = get_model_and_tokenizer(model_name_or_path)

    encoded_input = tokenizer(texts.tolist(), return_tensors="pt",
                              padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # consider attentiontin mean pooling
    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.numpy()

    result = embeddings_np.tolist()
    return result

# 3rd parties
import torch

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    """_summary_

    Args:
        model_output (_type_): _description_
        attention_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()

    # token_embeddings * input_mask_expanded: This multiplication effectively zeros out the embedding vectors for all padding tokens.
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 3rd parties
import torch
import torch.nn as nn
import torch.nn.functional as F

# local import
from model.mean_pooling import mean_pooling


class TripletLossModel(nn.Module):
    """
    A PyTorch model for computing triplet loss using a base transformer model.

    This model takes an anchor, positive, and negative input, computes their embeddings
    using a base transformer model, and calculates the triplet loss to ensure that
    the anchor is closer to the positive than to the negative by a specified margin.

    Args:
        nn.Module: Inherits from PyTorch's nn.Module.
    """

    def __init__(self, base_model):
        super(TripletLossModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask):

        def encode(input_ids, attention_mask):
            output = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pooling(output, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

        anchor_emb = encode(input_ids, attention_mask)
        pos_emb = encode(positive_input_ids, positive_attention_mask)
        neg_emb = encode(negative_input_ids, negative_attention_mask)

        # Calculate the triplet loss
        triplet_loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)

        return {"loss": triplet_loss}

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        # Compute pairwise distances
        positive_distance = F.pairwise_distance(
            anchor, positive, p=2).pow(2)  # Euclidean distance
        negative_distance = F.pairwise_distance(anchor, negative, p=2).pow(2)

        # Triplet loss function (hinge loss)
        # We want to minimize the loss, which is the difference between the positive and negative distances plus the margin. This means that we want the positive distance to be smaller than the negative distance by at least the margin.
        # The term positive_distance - negative_distance + margin is a margin-based constraint. The goal is:
        # Make the anchor and positive pair as close as possible (minimize positive_distance).
        # Ensure the anchor and negative pair are sufficiently far apart (maximize negative_distance).
        # When positive_distance - negative_distance + margin > 0, the loss is non-zero, and the model will be penalized if the anchor-negative pair is not sufficiently far from the anchor-positive pair.
        # The torch.clamp(..., min=0.0) ensures that the loss cannot be negative, which means that if the positive_distance is already smaller than negative_distance + margin, there is no penalty, and the model has achieved a good separation between the anchor-positive and anchor-negative pairs.

        # e.g.
        # If the positive_distance (anchor vs. positive) = 0.3, and negative_distance (anchor vs. negative) = 1.2, the loss will be:
        # In this case, the loss is zero, meaning the anchor and positive are sufficiently close, and the anchor and negative are sufficiently far apart.
        # loss=max(0.3−1.2+0.5,0)=max(−0.4,0)=0
        # e.g.
        # However, if the positive_distance = 0.8 and negative_distance = 0.9, the loss will b
        # loss=max(1.2−0.3+0.5,0)=max(1.4,0)=1.4

        loss = torch.clamp(positive_distance -
                           negative_distance + margin, min=0.0)
        return loss.mean()

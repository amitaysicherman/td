import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import core, tasks, layers, models, data, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        num_nodes = graph.num_residues if graph.view == "residue" else graph.num_nodes
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(torch.arange(len(num_nodes), device=self.device), num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        # Save the true residue types at masked positions
        target = graph.residue_type[node_index]

        # Get mask token id from tokenizer
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "mask_token_id"):
            mask_id = self.model.tokenizer.mask_token_id
        elif hasattr(self.model, "sequence_model") and hasattr(self.model.sequence_model, "alphabet"):
            mask_id = self.model.sequence_model.alphabet.get_idx("<mask>")
        else:
            raise AttributeError("Cannot find mask token id in model or sequence_model")

        # Mask the selected residues
        with graph.residue():
            graph.residue_type[node_index] = mask_id
            if hasattr(graph, "residue_feature"):
                graph.residue_feature[node_index] = 0

        # Prepare input for the model (input is residue_type)
        input = graph.residue_type

        # Forward pass
        output = self.model(graph, input)
        residue_feature = output["residue_feature"]

        # Get the features at the masked positions
        node_feature = residue_feature[node_index]

        # Predict residue types
        pred = self.mlp(node_feature)
        return pred, target



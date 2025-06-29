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
    inv_mapping = {v: k for k, v in data.Protein.residue_symbol2id.items()}

    def predict_and_target(self, batch, all_loss=None, metric=None):
        # Extract sequence and positions from the graph
        graph = batch["graph"]
        # sequence = graph.residue_type  # [N]
        print(type(self.model.model))
        seq_str=" ".join([self.inv_mapping[x.item()] for x in graph.residue_type])
        sequence = self.model.tokenizer(seq_str,return_tensors='pt')["input_ids"]  # Convert to token ids
        # sequence = self.model.apply_mapping(sequence)  # Convert to token ids
        positions = graph.node_position  # [N, 3]
        batch_size = graph.batch_size
        num_residues = graph.num_residues  # [batch]
        num_residues = num_residues + 2  # Add 2 for start and end tokens
        max_len = num_residues.max().item()

        # Pad sequence and positions to [batch, max_len]
        padded_sequence = torch.full((batch_size, max_len), self.model.tokenizer.pad_token_id, dtype=sequence.dtype,
                                     device=sequence.device)
        padded_positions = torch.zeros((batch_size, max_len, 3), dtype=positions.dtype, device=positions.device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=sequence.device)
        start = 0
        for i, n in enumerate(num_residues):
            end = start + n
            padded_sequence[i, :n] = sequence[start:end]
            padded_positions[i, :n] = positions[start:end]
            mask[i, :n] = 1
            start = end

        # Sample mask positions
        num_samples = (num_residues * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(torch.arange(batch_size, device=sequence.device), num_samples)
        node_index = (torch.rand(num_sample, device=sequence.device) * num_residues[sample2graph]).long()
        # node_index: indices within each sequence
        mask_indices = (sample2graph, node_index)
        target = padded_sequence[mask_indices]

        # Mask the selected residues
        mask_id = self.model.tokenizer.mask_token_id
        masked_sequence = padded_sequence.clone()
        masked_sequence[mask_indices] = mask_id

        # Forward
        output = self.model(masked_sequence, padded_positions)
        residue_feature = output["residue_feature"]  # [batch, max_len, ...]
        pred = residue_feature[mask_indices]

        return pred, target


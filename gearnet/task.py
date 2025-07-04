import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_min

from torchdrug import core, tasks, layers
from torchdrug.data import constant, Protein
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):
    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]

        # Define special token IDs consistent with TorchDrug's default ProteinBERT.
        AMINO_ACID_VOCAB_SIZE = constant.NUM_AMINO_ACID
        BOS_ID = AMINO_ACID_VOCAB_SIZE
        EOS_ID = AMINO_ACID_VOCAB_SIZE + 1
        PAD_ID = AMINO_ACID_VOCAB_SIZE + 2
        MASK_ID = AMINO_ACID_VOCAB_SIZE + 3

        # Separate original residue types and positions for each protein in the batch
        all_original_sequences_variadic = []
        all_positions_variadic = []
        current_node_idx = 0
        for num_res in graph.num_residues:
            all_original_sequences_variadic.append(graph.residue_type[current_node_idx: current_node_idx + num_res])
            all_positions_variadic.append(graph.node_position[current_node_idx: current_node_idx + num_res])
            current_node_idx += num_res

        # --- Consolidated Masking Logic ---
        # These lists will store the target values and the indices for extracting predictions.
        target_values = []
        final_mask_indices_batch = []
        final_mask_indices_token = []

        # Determine the maximum padded length across the batch (for input_ids and positions)
        max_padded_length = 0
        if len(all_original_sequences_variadic) > 0:
            max_padded_length = max(len(s) for s in all_original_sequences_variadic) + 2  # +2 for BOS/EOS

        padded_sequences_list = []
        padded_positions_list = []

        # Iterate through each protein to prepare inputs and collect targets
        current_res_offset_in_graph = 0  # Offset in the flat graph.residue_type tensor
        for i, num_res in enumerate(graph.num_residues):
            # Calculate how many residues to mask for the current protein
            num_to_mask = (num_res * self.mask_rate).long().clamp(1)

            # IMPORTANT: Generate random indices ONCE for this protein.
            # These indices are relative to the *original* residues of this protein.
            indices_to_mask_in_protein = torch.randperm(num_res, device=self.device)[:num_to_mask]

            # 1. Collect target values: these are the original residue IDs at the masked positions
            original_values_for_this_protein = all_original_sequences_variadic[i]
            target_values.append(original_values_for_this_protein[indices_to_mask_in_protein])

            # 2. Prepare `input_ids` with BOS/EOS tokens for the current protein
            current_seq_with_special_tokens = torch.cat([
                torch.tensor([BOS_ID], device=self.device),  # Prepend BOS
                original_values_for_this_protein,
                torch.tensor([EOS_ID], device=self.device)  # Append EOS
            ])
            # Pad the sequence to `max_padded_length`
            padding_len = max_padded_length - len(current_seq_with_special_tokens)
            padded_seq = F.pad(current_seq_with_special_tokens, (0, padding_len), value=PAD_ID)
            padded_sequences_list.append(padded_seq)

            # 3. Prepare `struct_positions` for the current protein (align with sequence padding)
            current_pos_variadic = all_positions_variadic[i]
            # Add zero-filled positions for BOS/EOS tokens
            bos_pos = torch.zeros(1, 3, device=self.device)
            eos_pos = torch.zeros(1, 3, device=self.device)
            current_pos_with_special_tokens = torch.cat([bos_pos, current_pos_variadic, eos_pos], dim=0)
            # Pad positions to match the sequence padding
            padded_pos = F.pad(current_pos_with_special_tokens, (0, 0, 0, padding_len), value=0.0)
            padded_positions_list.append(padded_pos)

            # 4. Store indices for masking `input_ids` and extracting `pred`
            # The indices for the padded sequence need to account for the BOS token at index 0.
            # So, an original residue index `j` becomes `j + 1` in the padded sequence.
            token_indices_for_current_protein = indices_to_mask_in_protein + 1

            # Store the batch index (`i`) and the token index for each masked position
            final_mask_indices_batch.append(torch.full_like(token_indices_for_current_protein, i))
            final_mask_indices_token.append(token_indices_for_current_protein)

            current_res_offset_in_graph += num_res  # Update offset for the next protein

        # Concatenate all target values into a single tensor
        target = torch.cat(target_values) if target_values else torch.empty(0, dtype=torch.long, device=self.device)

        # Stack lists of tensors to create batch tensors for model input
        input_ids = torch.stack(padded_sequences_list) if padded_sequences_list else torch.empty(0, max_padded_length,
                                                                                                 dtype=torch.long,
                                                                                                 device=self.device)
        positions = torch.stack(padded_positions_list) if padded_positions_list else torch.empty(0, max_padded_length,
                                                                                                 3, dtype=torch.float,
                                                                                                 device=self.device)

        # Apply the MASK_ID to the `input_ids` at the designated masked positions
        if final_mask_indices_batch:
            final_mask_indices_batch_tensor = torch.cat(final_mask_indices_batch)
            final_mask_indices_token_tensor = torch.cat(final_mask_indices_token)

            # This tuple is used to index both input_ids for masking and output.logits for prediction
            final_mask_indices_tuple = (final_mask_indices_batch_tensor, final_mask_indices_token_tensor)

            # Apply mask to input_ids (changing the token to MASK_ID)
            input_ids[final_mask_indices_tuple] = MASK_ID
        else:
            # If no masks, create empty index tensors
            final_mask_indices_tuple = (
            torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device))

        # --- Forward Pass through the Model ---
        output = self.model(graph=graph, input_ids=input_ids, struct_positions=positions)

        # Get predictions (logits) only for the masked positions using the exact same indices
        if final_mask_indices_tuple[0].numel() > 0:
            pred = output['logits'][final_mask_indices_tuple]
        else:
            # If no positions were masked, ensure pred is also empty and matches target's potential emptiness
            # The output dimension for logits is the vocabulary size (self.model.input_dim)
            pred = torch.empty(0, self.model.input_dim, device=self.device)

        return pred, target

    def evaluate(self, pred, target):
        metric = {}
        if pred.numel() > 0 and target.numel() > 0:
            accuracy = (pred.argmax(dim=-1) == target).float().mean()
            name = tasks._get_metric_name("acc")
            metric[name] = accuracy
        else:
            metric[tasks._get_metric_name("acc")] = torch.tensor(0.0, device=self.device)

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        if pred.numel() > 0 and target.numel() > 0:
            loss = F.cross_entropy(pred, target)
            name = tasks._get_criterion_name("ce")
            metric[name] = loss
            all_loss += loss
        else:
            metric[tasks._get_criterion_name("ce")] = torch.tensor(0.0, device=self.device)

        metric.update(self.evaluate(pred, target))

        return all_loss, metric
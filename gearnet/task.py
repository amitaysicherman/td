import torch
from torchdrug import core, tasks, data
from torchdrug.core import Registry as R


@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):
    inv_mapping = {v: k for k, v in data.Protein.residue_symbol2id.items()}

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        all_sequences = []
        all_positions = []
        start=0
        for i, n in enumerate(graph.num_residues):
            seq_str = " ".join([self.inv_mapping[x.item()] for x in graph.residue_type[start:start+n]])
            sequence = self.model.tokenizer(seq_str, return_tensors='pt')["input_ids"].flatten()
            all_sequences.append(sequence)
            pos = graph.node_position[start:start+n]
            all_positions.append(pos)
            start += n
        sequence = torch.nn.utils.rnn.pad_sequence(all_sequences, batch_first=True, padding_value=self.model.tokenizer.pad_token_id)
        positions = torch.nn.utils.rnn.pad_sequence(all_positions, batch_first=True, padding_value=0.0)

        num_samples = (graph.num_residues * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(torch.arange(graph.batch_size, device=sequence.device), num_samples)
        node_index = (torch.rand(num_sample, device=sequence.device) * graph.num_residues[sample2graph]).long()
        mask_indices = (sample2graph, node_index)
        target = sequence[mask_indices]

        # Mask the selected residues
        mask_id = self.model.tokenizer.mask_token_id
        masked_sequence = sequence.clone()
        masked_sequence[mask_indices] = mask_id


        # Forward
        output = self.model(masked_sequence, positions)
        pred = output.logits[mask_indices]

        return pred, target

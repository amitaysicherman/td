import os
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="/Users/amitay.s/PycharmProjects/scratch/protein-model-weights/prot_bert_bfd/")
BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="/Users/amitay.s/PycharmProjects/scratch/protein-model-weights/prot_bert_bfd/")

@R.register("models.DualPositionalTransformer")
class DualPositionalTransformer(nn.Module, core.Configurable):
    """
    Transformer with dual positional embeddings: sequence and structure-based (3D).
    """

    def __init__(self, path, emb_dim=1024):
        super().__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path


        # Load Bert/ESM model and tokenizer
        model, tokenizer = self.load_weight(path)
        mapping = self.construct_mapping(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.register_buffer("mapping", mapping)

        # Sequence positional embedding (reuse from model)
        self.seq_pos_emb = self.model.embeddings.position_embeddings

        # Structure positional embedding: project 3D coords to emb_dim
        self.struct_pos_proj = nn.Linear(3, emb_dim)

        self.emb_dim = emb_dim
        self.output_dim = emb_dim

    def load_weight(self, path):
        # You can swap BertModel for ESM if you have it
        tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="/Users/amitay.s/PycharmProjects/scratch/protein-model-weights/prot_bert_bfd/")
        model=BertModel.from_pretrained("Rostlab/prot_bert_bfd", cache_dir="/Users/amitay.s/PycharmProjects/scratch/protein-model-weights/prot_bert_bfd/")

        # tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        # model = BertModel.from_pretrained(model_path)
        return model, tokenizer

    def construct_mapping(self, tokenizer):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = tokenizer._convert_token_to_id(token)
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        # Ensure model is on the same device as input
        device = input.device
        self.model = self.model.to(device)

        # Prepare input ids\

        input = graph.residue_type
        input = self.mapping[input]
        size = graph.num_residues
        size_ext = size
        bos = torch.ones(graph.batch_size, dtype=torch.long, device=device) * self.tokenizer.cls_token_id
        input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=device) * self.tokenizer.sep_token_id
        input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.tokenizer.pad_token_id)[0]

        # Get 3D positions (pad as needed)
        node_position = graph.node_position  # [N, 3]

        # Get sequence positional ids
        seq_pos_ids = torch.arange(input.size(1), device=device).unsqueeze(0).expand(input.size(0), -1)

        # Get sequence positional embedding
        seq_pos_emb = self.seq_pos_emb(seq_pos_ids)

        # Project 3D positions to embedding space
        struct_pos_emb = self._get_struct_pos_emb(graph, size_ext, input.shape, node_position, device)

        # Get token embedding
        token_emb = self.model.embeddings.word_embeddings(input)

        # Combine embeddings
        emb = token_emb + seq_pos_emb + struct_pos_emb

        # Create proper attention mask for BERT
        # BERT expects attention_mask of shape (batch_size, seq_len) where 1 = attend, 0 = ignore
        attention_mask = (input != self.tokenizer.pad_token_id).long()
        # Use BERT's forward method instead of directly calling encoder
        # This ensures proper attention mask handling
        outputs = self.model(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        
        # Convert to variadic
        residue_feature = functional.padded_to_variadic(sequence_output, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        starts = starts + 1  # Skip CLS token
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]

        return {
            "residue_feature": residue_feature
        }

    def _get_struct_pos_emb(self, graph, size_ext, input_shape, node_position, device):
        batch_size, seq_len = input_shape
        struct_pos_emb = torch.zeros(batch_size, seq_len, self.emb_dim, device=device)
        starts = size_ext.cumsum(0) - size_ext
        for i, (start, length) in enumerate(zip(starts, size_ext)):
            n_res = length.item()
            if n_res == 0:
                continue
            # Defensive: only fill up to the available space in both tensors
            max_fill = min(n_res, struct_pos_emb.shape[1] - 2, node_position.shape[0] - start)
            if max_fill > 0:
                struct_pos_emb[i, 1:1+max_fill] = self.struct_pos_proj(node_position[start:start+max_fill])
        return struct_pos_emb

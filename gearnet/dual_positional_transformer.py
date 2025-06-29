import os
import torch
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM
from torchdrug import core, data
from torchdrug.core import Registry as R


class DualPositionalBertEmbeddings(nn.Module):
    """
    Bert Embeddings with optional structure-based (3D) positional embedding.
    """

    def __init__(self, orig_embeddings, emb_dim):
        super().__init__()
        # Copy all original embedding layers
        self.word_embeddings = orig_embeddings.word_embeddings
        self.position_embeddings = orig_embeddings.position_embeddings
        self.token_type_embeddings = orig_embeddings.token_type_embeddings
        self.LayerNorm = orig_embeddings.LayerNorm
        self.dropout = orig_embeddings.dropout
        self.position_embedding_type = getattr(orig_embeddings, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", orig_embeddings.position_ids.clone(), persistent=False)
        self.register_buffer("token_type_ids", orig_embeddings.token_type_ids.clone(), persistent=False)
        # Add structure positional projection
        self.struct_pos_proj = nn.Linear(3, emb_dim)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
            struct_positions=None,  # [batch, seq_len, 3]
            mode=0,  # 0: seq only, 1: struct only, 2: sum
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # Add structure positional embedding if provided
        if struct_positions is not None:
            struct_pos_emb = self.struct_pos_proj(struct_positions)
            if mode == 0:
                pass  # only sequence embedding
            elif mode == 1:
                embeddings = struct_pos_emb
            elif mode == 2:
                embeddings[:, 1:-1, :] += struct_pos_emb  # [1:-1] to avoid adding to special tokens
            else:
                raise ValueError("Invalid mode for dual positional embedding")
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@R.register("models.DualPositionalTransformer")
class DualPositionalTransformer(nn.Module, core.Configurable):
    """
    Transformer with dual positional embeddings: sequence and structure-based (3D).
    """

    def __init__(self, path, emb_dim=1024, mode=0):
        super().__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        # Load Bert/ESM model and tokenizer
        model, tokenizer = self.load_weight(path)
        self.model = model
        self.tokenizer = tokenizer
        self.mapping, self.inverted_mapping = self.construct_mappings(tokenizer)
        self.emb_dim = emb_dim
        self.output_dim = emb_dim
        self.mode = mode

        # Replace Bert embeddings with dual positional embedding
        self.model.bert.embeddings = DualPositionalBertEmbeddings(self.model.bert.embeddings, emb_dim)

    def load_weight(self, path):
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", cache_dir=path)
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", cache_dir=path)
        return model, tokenizer

    def construct_mappings(self, tokenizer):
        mapping = {}
        for res, id_ in data.Protein.residue_symbol2id.items():
            mapping[id_] = tokenizer._convert_token_to_id(res.upper())
        inverted_mapping = {token: i for i, token in mapping.items()}
        return mapping, inverted_mapping

    def apply_mapping(self, input):
        for k, v in self.mapping.items():
            input = torch.where(input == k, v, input)
        return input

    def forward(self, input, positions):
        """
        input: [batch, seq_len] (token ids)
        positions: [batch, seq_len, 3]
        """
        # Attention mask
        attention_mask = (input != self.tokenizer.pad_token_id).long()
        return self.model(
            input_ids=input,
            attention_mask=attention_mask,
            position_ids=None,
            token_type_ids=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            struct_positions=positions,
            mode=self.mode,
        )

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers
from torchdrug.data import constant  # Import constant to get NUM_AMINO_ACID
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.DualPositionalTransformer")
class DualPositionalTransformer(nn.Module, core.Configurable):
    """
    Protein BERT proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension (typically vocabulary size: num_amino_acids + special tokens)
        hidden_dim (int, optional): hidden dimension
        num_layers (int, optional): number of Transformer blocks (implemented via `ProteinBERTBlock` layers)
        num_heads (int, optional): number of attention heads
        intermediate_dim (int, optional): intermediate hidden dimension of Transformer block
        activation (str or function, optional): activation function
        hidden_dropout (float, optional): dropout ratio of hidden features
        attention_dropout (float, optional): dropout ratio of attention maps
        max_position (int, optional): maximum number of positions for absolute positional embeddings
        use_struct_embedding (bool, optional): whether to use structural positional embedding
        struct_embedding_mode (int, optional): Mode for combining embeddings:
                                              0: sequence only,
                                              1: structure only,
                                              2: sum (default)
    """

    def __init__(self, input_dim, hidden_dim=768, num_layers=12, num_heads=12, intermediate_dim=3072,
                 activation="gelu", hidden_dropout=0.1, attention_dropout=0.1, max_position=8192,
                 use_struct_embedding=False, struct_embedding_mode=2):
        super(DualPositionalTransformer, self).__init__()
        self.input_dim = input_dim  # This is the vocab size
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position = max_position
        self.use_struct_embedding = use_struct_embedding
        self.struct_embedding_mode = struct_embedding_mode

        # The embedding layer for token IDs (residues + special tokens like BOS, EOS, PAD, MASK)
        self.embedding = nn.Embedding(input_dim, hidden_dim)

        # Standard absolute positional embedding, independent of structure
        self.position_embedding = nn.Embedding(max_position, hidden_dim)

        if self.use_struct_embedding:
            # Linear layer to project 3D coordinates (x, y, z) to the model's hidden dimension
            self.struct_pos_proj = nn.Linear(3, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        # These are the core Transformer encoder blocks. `ProteinBERT` uses `layers.ProteinBERTBlock`
        # to implement its multi-head attention and feed-forward network layers.
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(layers.ProteinBERTBlock(hidden_dim, intermediate_dim, num_heads,
                                                       attention_dropout, hidden_dropout, activation))

        # Output layer for the graph-level representation (typically derived from the [CLS] token)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

        # Classification Head for Masked Residue Prediction
        # This head projects the hidden states back to the vocabulary size to predict masked tokens.
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),  # Use GELU as is common in BERT models
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)  # Project to vocabulary size (vocab_size == input_dim)
        )

    # Updated forward method signature and logic
    def forward(self, graph, input_ids, struct_positions=None, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): A `torchdrug.data.Protein` object representing the batch of proteins.
            input_ids (Tensor): Padded input token IDs, already including BOS/EOS and padding.
                                Shape: [batch_size, padded_sequence_length].
            struct_positions (Tensor, optional): Padded 3D coordinates corresponding to `input_ids`.
                                                 Shape: [batch_size, padded_sequence_length, 3].
                                                 Expected to include placeholder coordinates (e.g., zeros)
                                                 for special tokens and padding tokens.
            all_loss (Tensor, optional): If specified, additional losses (e.g., auxiliary losses from the model itself)
                                         can be added to this tensor.
            metric (dict, optional): If specified, metrics computed internally by the model can be added to this dict.

        Returns:
            dict with ``residue_feature``, ``graph_feature``, and ``logits`` fields:
                - ``residue_feature``: A tensor of residue representations of shape :math:`(|V_{res}|, d)`,
                                       where :math:`|V_{res}|` is the total number of non-special, non-padded residues
                                       across the batch, and :math:`d` is `hidden_dim`.
                - ``graph_feature``: A tensor of graph representations of shape :math:`(n, d)`,
                                     where :math:`n` is the number of proteins in the batch.
                - ``logits``: Raw prediction scores for each token in the vocabulary at each position.
                              Shape: [batch_size, padded_sequence_length, input_dim (vocab_size)].
        """
        # Determine the padding token ID for mask generation.
        # This aligns with the PAD_ID used in ResidueTypePrediction task: constant.NUM_AMINO_ACID + 2
        pad_token_id_for_mask = constant.NUM_AMINO_ACID + 2

        # Create an attention mask: 1 for actual tokens (and special tokens like BOS/EOS), 0 for padding tokens
        # The task ensures input_ids are already padded correctly.
        attention_mask = (input_ids != pad_token_id_for_mask).long()  # [batch_size, padded_sequence_length]

        # Get sequence embeddings from token IDs
        sequence_embeddings = self.embedding(input_ids)  # [batch_size, padded_sequence_length, hidden_dim]

        # Add standard absolute positional embeddings
        position_indices = torch.arange(input_ids.shape[1], device=input_ids.device)  # [padded_sequence_length]
        position_embeddings = self.position_embedding(position_indices).unsqueeze(
            0)  # [1, padded_sequence_length, hidden_dim]

        # Combine sequence and absolute positional embeddings
        embeddings = sequence_embeddings + position_embeddings  # [batch_size, padded_sequence_length, hidden_dim]

        # Integrate structural positional embedding if enabled
        if self.use_struct_embedding and struct_positions is not None:
            # Validate that struct_positions has the same batch and sequence dimensions as input_ids
            if struct_positions.shape[:2] != input_ids.shape:
                raise ValueError(
                    f"Structural positions shape {struct_positions.shape} mismatch with input_ids shape {input_ids.shape}. "
                    f"Expected struct_positions to be [batch_size, padded_sequence_length, 3]. "
                    f"Ensure struct_positions are padded and aligned with input_ids.")

            # Project 3D coordinates to the embedding dimension
            struct_pos_emb = self.struct_pos_proj(struct_positions)  # [batch_size, padded_sequence_length, hidden_dim]

            # Apply the chosen combination mode
            if self.struct_embedding_mode == 0:
                pass  # Only sequence embedding (already `embeddings` value)
            elif self.struct_embedding_mode == 1:
                embeddings = struct_pos_emb  # Replace with only structural embedding
            elif self.struct_embedding_mode == 2:
                embeddings += struct_pos_emb  # Sum sequence and structural embeddings
            else:
                raise ValueError(
                    "Invalid mode for dual positional embedding. Must be 0 (seq only), 1 (struct only), or 2 (sum).")

        # Apply layer normalization and dropout to the combined embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Prepare mask for Transformer blocks: [batch_size, padded_sequence_length, 1]
        # TorchDrug's ProteinBERTBlock expects this mask format.
        mask_for_transformer = attention_mask.unsqueeze(-1)

        # Pass through the Transformer encoder layers (`ProteinBERTBlock`s)
        for layer in self.layers:
            embeddings = layer(embeddings, mask_for_transformer)

        # Compute Logits for Masked Residue Prediction
        # The classification head takes the contextualized embeddings and predicts the vocabulary.
        logits = self.cls_head(embeddings)  # [batch_size, padded_sequence_length, input_dim (vocab_size)]

        # Extract residue-level features and graph-level features
        # The task prepares `input_ids` with BOS at index 0 and EOS at `num_res + 1`.
        # Actual residues are at indices from 1 to `num_res`.

        residue_features_list = []
        for i, num_res in enumerate(graph.num_residues):
            # `graph.num_residues[i]` gives the original number of amino acids for the i-th protein.
            # Extract features corresponding to these actual residues, ignoring [CLS]/BOS, [SEP]/EOS, and padding.
            residue_features_list.append(embeddings[i, 1: 1 + num_res, :])

        # Concatenate all extracted residue features into a single tensor
        residue_feature = torch.cat(residue_features_list, dim=0)

        # The graph-level feature is typically taken from the [CLS]/BOS token's embedding (index 0)
        graph_feature = embeddings[:, 0]  # [batch_size, hidden_dim]
        graph_feature = self.linear(graph_feature)
        graph_feature = F.tanh(graph_feature)  # Apply activation

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
            "logits": logits  # Return raw prediction scores for the task
        }
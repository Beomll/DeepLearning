import torch
import torch.nn as nn
import pickle


class BERT(nn.Module):
    def __init__(self, vocab_size, args):
        super().__init__()

        self.hidden_size = args.hidden_size

        # embedding layer
        self.tok_emb = nn.Embedding(vocab_size, self.hidden_size, padding_idx=0)  # pad token을 위한 padding_idx=0 처리
        self.seg_emb = nn.Embedding(args.segment_type, self.hidden_size)
        self.pos_emb = nn.Embedding(args.max_len, self.hidden_size)

        # transformer layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=args.attn_heads,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=args.transformer_layers
        )

    def forward(self, inputs, positions, segments, attn_mask):
        x = self.tok_emb(inputs), self.pos_emb(positions) + self.seg_emb(segments)
        x = nn.Dropout(p=0.1)(x)
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        return x


# Language Model
"""
    Masked Language Model + Next Sentence Prediction Model
"""
class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()

        self.bert = bert

        # linear layer for Language Model Prediction
        self.next_sentence = nn.Linear(self.bert.hidden_size, 2)
        self.mask_lm = nn.Linear(self.bert.hidden_size, vocab_size)

    def forward(self, inputs, positions, segments, attn_mask):
        x = self.bert(inputs, positions, segments, attn_mask)
        return self.next_sentence(x[:, 0, :]), self.mask_lm(x)










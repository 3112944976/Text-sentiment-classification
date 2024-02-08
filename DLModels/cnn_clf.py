import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CNNConfig


class CNN_Classifier(nn.Module):
    def __init__(self, vocab_size):
        super(CNN_Classifier, self).__init__()
        self.emb_size = CNNConfig.emb_size
        self.num_filters = CNNConfig.num_filters
        self.window_sizes = CNNConfig.window_sizes
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, [window_size,
                                            self.emb_size], padding=(window_size-1, 0))
            for window_size in self.window_sizes
        ])
        self.fc = nn.Linear(self.num_filters * len(self.window_sizes), 1)

    def forward(self, sentences):
        embed_sents = self.embedding(sentences)
        embed_sents = embed_sents.unsqueeze(1)
        sent_features = []
        for conv in self.convs:
            conv_out = F.relu(conv(embed_sents))
            conv_out = conv_out.squeeze(-1)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            sent_features.append(pool_out)
        sent_features = torch.cat(sent_features, dim=1)
        logits = self.fc(sent_features)
        return logits

    def init_embedding(self, embedding):
        self.embedding.weight.data = embedding

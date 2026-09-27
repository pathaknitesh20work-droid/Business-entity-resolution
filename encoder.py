import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import VOCAB


class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        output_dim=128
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                embedding_dim,
                64,
                kernel_size=3
            ),

            nn.Conv1d(
                embedding_dim,
                64,
                kernel_size=4
            ),

            nn.Conv1d(
                embedding_dim,
                64,
                kernel_size=5
            )
        ])

        self.fc = nn.Linear(
            64 * 3,
            output_dim
        )

    def forward(self, x):

        x = self.embedding(x)

        x = x.transpose(1, 2)

        outputs = []

        for conv in self.convs:

            h = F.relu(conv(x))

            h = F.max_pool1d(
                h,
                kernel_size=h.size(2)
            ).squeeze(2)

            outputs.append(h)

        x = torch.cat(outputs, dim=1)

        return self.fc(x)


class BusinessEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=len(VOCAB),
        embedding_dim=64,
        output_dim=256
    ):

        super().__init__()

        self.name_encoder = TextCNN(
            vocab_size,
            embedding_dim,
            128
        )

        self.address_encoder = TextCNN(
            vocab_size,
            embedding_dim,
            128
        )

        self.country_encoder = TextCNN(
            vocab_size,
            embedding_dim,
            32
        )

        self.fc = nn.Sequential(

            nn.Linear(
                128 + 128 + 32,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                output_dim
            )
        )

    def forward(
        self,
        name,
        address,
        country
    ):

        name_vec = self.name_encoder(name)

        address_vec = self.address_encoder(address)

        country_vec = self.country_encoder(country)

        combined = torch.cat(
            [
                name_vec,
                address_vec,
                country_vec
            ],
            dim=1
        )

        embedding = self.fc(combined)

        return F.normalize(
            embedding,
            p=2,
            dim=1
        )
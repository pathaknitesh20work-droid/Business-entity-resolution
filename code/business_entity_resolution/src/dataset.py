import torch
from torch.utils.data import Dataset

from normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)

from tokenizer import encode_text


class BusinessDataset(Dataset):

    def __init__(self, dataframe):

        self.data = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        name = normalize_name(row["business_name"])
        address = normalize_address(row["business_address"])
        country = normalize_country(row["country"])

        return {
            "name": torch.tensor(
                encode_text(name),
                dtype=torch.long
            ),

            "address": torch.tensor(
                encode_text(address),
                dtype=torch.long
            ),

            "country": torch.tensor(
                encode_text(country, max_length=32),
                dtype=torch.long
            ),

            "entity_id": row["entity_id"]
        }
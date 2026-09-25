import torch

from torch.utils.data import Dataset

from tokenizer import encode_text


class PairDataset(Dataset):
    """
    PyTorch Dataset for Siamese encoder training.

    The supplied source DataFrames must already contain:

        norm_name
        norm_address
        norm_country
    """


    def __init__(
        self,
        pairs,
        source1_lookup,
        candidate_lookup
    ):

        self.pairs = (
            pairs
            .reset_index(drop=True)
        )

        self.source1_lookup = (
            source1_lookup
        )

        self.candidate_lookup = (
            candidate_lookup
        )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.pairs
        )


    # ========================================================
    # ENTITY → TENSORS
    # ========================================================

    @staticmethod
    def encode_entity(row):

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        name_tensor = torch.tensor(
            encode_text(
                row["norm_name"]
            ),
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        address_tensor = torch.tensor(
            encode_text(
                row["norm_address"]
            ),
            dtype=torch.long
        )


        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        country_tensor = torch.tensor(
            encode_text(
                row["norm_country"],
                max_length=32
            ),
            dtype=torch.long
        )


        return (
            name_tensor,
            address_tensor,
            country_tensor
        )


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        pair = self.pairs.iloc[
            index
        ]


        source1_id = pair[
            "source1_entity_id"
        ]

        candidate_id = pair[
            "candidate_entity_id"
        ]

        label = float(
            pair["label"]
        )


        # ----------------------------------------------------
        # Retrieve already-normalized entities
        # ----------------------------------------------------

        source1_row = (
            self.source1_lookup.loc[
                source1_id
            ]
        )

        candidate_row = (
            self.candidate_lookup.loc[
                candidate_id
            ]
        )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        (
            name_a,
            address_a,
            country_a
        ) = self.encode_entity(
            source1_row
        )


        (
            name_b,
            address_b,
            country_b
        ) = self.encode_entity(
            candidate_row
        )


        return {
            "name_a": name_a,

            "address_a": address_a,

            "country_a": country_a,

            "name_b": name_b,

            "address_b": address_b,

            "country_b": country_b,

            "label": torch.tensor(
                label,
                dtype=torch.float32
            )
        }
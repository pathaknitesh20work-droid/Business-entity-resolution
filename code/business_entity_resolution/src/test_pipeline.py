from data_loader import load_training_data
from normalization import (
    normalize_name,
    normalize_address,
    normalize_country
)
from tokenizer import encode_text
from dataset import BusinessDataset
from encoder import BusinessEncoder

import torch


print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

source1, source2, source3, ground_truth = load_training_data()

print("Source 1 shape:", source1.shape)
print("Source 2 shape:", source2.shape)
print("Source 3 shape:", source3.shape)
print("Ground Truth shape:", ground_truth.shape)

print("\nSource 1 columns:")
print(source1.columns.tolist())

print("\nGround Truth columns:")
print(ground_truth.columns.tolist())


print("\n" + "=" * 70)
print("STEP 2: CHECKING NORMALIZATION")
print("=" * 70)

sample_name = source1.iloc[0]["business_name"]
sample_address = source1.iloc[0]["business_address"]
sample_country = source1.iloc[0]["country"]

print("Original name:", sample_name)
print("Normalized name:", normalize_name(sample_name))

print("\nOriginal address:", sample_address)
print("Normalized address:", normalize_address(sample_address))

print("\nOriginal country:", sample_country)
print("Normalized country:", normalize_country(sample_country))


print("\n" + "=" * 70)
print("STEP 3: CHECKING TOKENIZATION")
print("=" * 70)

encoded_name = encode_text(normalize_name(sample_name))

print("Encoded name:")
print(encoded_name)

print("Encoded length:", len(encoded_name))


print("\n" + "=" * 70)
print("STEP 4: CHECKING DATASET")
print("=" * 70)

dataset =  BusinessDataset(source1)

print("Dataset length:", len(dataset))

sample = dataset[0]

print("\nSample keys:")
print(sample.keys())

for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(
            key,
            "shape =", tuple(value.shape),
            "dtype =", value.dtype
        )
    else:
        print(key, "=", value)


print("\n" + "=" * 70)
print("STEP 5: CHECKING NEURAL NETWORK")
print("=" * 70)

model = BusinessEncoder()

print(model)

print("\nRunning one sample through encoder...")

# Add batch dimension
name = sample["name"].unsqueeze(0)
address = sample["address"].unsqueeze(0)
country = sample["country"].unsqueeze(0)

with torch.no_grad():
    embedding = model(name, address, country)

print("Embedding shape:", embedding.shape)
print("Embedding dtype:", embedding.dtype)

print("\nFirst 10 embedding values:")
print(embedding[0][:10])

print("\nEmbedding norm:")
print(torch.norm(embedding, dim=1))

print("\n" + "=" * 70)
print("ALL SANITY CHECKS COMPLETED")
print("=" * 70)
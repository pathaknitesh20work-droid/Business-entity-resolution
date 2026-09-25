import string


SPECIAL_TOKENS = ["<PAD>", "<UNK>"]


def build_vocab():

    chars = list(
        string.ascii_lowercase
        + string.digits
        + " "
    )

    vocab = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    for char in chars:
        if char not in vocab:
            vocab[char] = len(vocab)

    return vocab


VOCAB = build_vocab()


def encode_text(text, max_length=256):

    ids = []

    for char in text[:max_length]:

        if char in VOCAB:
            ids.append(VOCAB[char])
        else:
            ids.append(VOCAB["<UNK>"])

    while len(ids) < max_length:
        ids.append(VOCAB["<PAD>"])

    return ids
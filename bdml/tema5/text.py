from __future__ import annotations


class CharTokenizer:
    def __init__(self):
        self.blank_token = "<blank>"
        self.characters = " abcdefghijklmnopqrstuvwxyz'"
        self.vocab = [self.blank_token] + list(self.characters)
        self.char_to_index = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.index_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.blank_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        normalized = text.lower().strip()
        indices: list[int] = []
        for char in normalized:
            if char in self.char_to_index:
                indices.append(self.char_to_index[char])
            else:
                indices.append(self.char_to_index[" "])
        return indices

    def decode_ctc(self, token_ids: list[int]) -> str:
        decoded_chars: list[str] = []
        previous = None
        for token_id in token_ids:
            if token_id == self.blank_id:
                previous = token_id
                continue
            if token_id == previous:
                continue
            char = self.index_to_char.get(token_id, " ")
            if char != self.blank_token:
                decoded_chars.append(char)
            previous = token_id
        return "".join(decoded_chars).strip()

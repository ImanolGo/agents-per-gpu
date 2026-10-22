"""Tokenizers for tests: one token per whitespace-separated word."""


class WordTokenizer:
    def count(self, text: str) -> int:
        return len(text.split())

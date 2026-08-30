import re
from typing import List
from .markov_chain import MarkovChain
# Splits text into sentences on ./!/? followed by whitespace, while
# keeping things reasonably simple (good enough for demo-quality corpora).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")
class WordMarkovModel:
    """Generates new sentences by learning word-to-word transition probabilities."""
    def __init__(self, order: int = 2, seed: int = None):
        """
        Args:
            order: Number of previous words used as context. order=1 is
                   very chaotic; order=2 or 3 tends to produce the most
                   entertaining "almost sensible" sentences.
            seed:  Optional random seed for reproducibility.
        """
        self.order = order
        self.chain = MarkovChain(order=order, seed=seed)
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    def train(self, text: str) -> "WordMarkovModel":
        """
        Train the model on a block of text. The text is split into
        sentences so that generated output always starts at a genuine
        sentence beginning rather than mid-sentence.
        """
        for sentence in self._split_sentences(text):
            words = _WORD_RE.findall(sentence)
            self.chain.train(words, is_start_of_sequence=True)
        return self
    def generate_sentence(self, max_words: int = 30) -> str:
        """Generate a single sentence (up to max_words long)."""
        words = self.chain.generate(length=max_words)
        return " ".join(words)
    def generate_text(self, num_sentences: int = 3, max_words_per_sentence: int = 30) -> str:
        """Generate several sentences and join them into a short paragraph."""
        sentences = [self.generate_sentence(max_words_per_sentence) for _ in range(num_sentences)]
        return " ".join(sentences)
    def stats(self) -> dict:
        return self.chain.stats()

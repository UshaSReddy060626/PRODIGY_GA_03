import re
from typing import List
from .markov_chain import MarkovChain
_WORD_RE = re.compile(r"[A-Za-z']+")
# A token marking the end of a word, so the chain learns when words
# naturally terminate instead of running on forever.
END = "\0"
class CharMarkovModel:
    """Generates new, invented words by learning character-to-character transitions."""
    def __init__(self, order: int = 3, seed: int = None):
        """
        Args:
            order: Number of previous characters used as context. order=3-4
                   is a good sweet spot: high enough that generated words
                   are pronounceable, low enough that they're genuinely new.
            seed:  Optional random seed for reproducibility.
        """
        self.order = order
        self.chain = MarkovChain(order=order, seed=seed)
    def train(self, text: str) -> "CharMarkovModel":
        """
        Train on a block of text. Each whitespace-delimited word becomes
        one training sequence (with an explicit END marker appended so
        the model learns realistic word lengths).
        """
        words = _WORD_RE.findall(text.lower())
        for word in words:
            if len(word) <= self.order:
                continue
            self.chain.train(list(word) + [END], is_start_of_sequence=True)
        return self
    def generate_word(self, max_length: int = 20) -> str:
        """Generate a single new word (stops early if the END token is produced)."""
        tokens: List[str] = self.chain.generate(length=max_length)
        chars = []
        for t in tokens:
            if t == END:
                break
            chars.append(t)
        return "".join(chars)
    def generate_words(self, count: int = 10, max_length: int = 20) -> List[str]:
        """Generate several new words, deduplicated against the training vocabulary is NOT done -- pure model output."""
        return [self.generate_word(max_length) for _ in range(count)]
    def stats(self) -> dict:
        return self.chain.stats()

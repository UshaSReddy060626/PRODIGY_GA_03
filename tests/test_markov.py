import os
import sys
import unittest
# Make the package importable when running tests directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markov_gen import MarkovChain, WordMarkovModel, CharMarkovModel
class TestMarkovChainCore(unittest.TestCase):
    def test_train_and_generate_basic(self):
        chain = MarkovChain(order=1, seed=42)
        chain.train(["a", "b", "a", "b", "a", "c"])
        result = chain.generate(length=5)
        self.assertEqual(len(result), 5)
        # every token generated must be one the model actually saw
        for token in result:
            self.assertIn(token, {"a", "b", "c"})
    def test_rejects_invalid_order(self):
        with self.assertRaises(ValueError):
            MarkovChain(order=0)
    def test_generate_before_train_raises(self):
        chain = MarkovChain(order=2)
        with self.assertRaises(RuntimeError):
            chain.generate(length=10)
    def test_reproducible_with_seed(self):
        text_tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "ran"]
        chain_a = MarkovChain(order=1, seed=7)
        chain_a.train(text_tokens)
        chain_b = MarkovChain(order=1, seed=7)
        chain_b.train(text_tokens)
        self.assertEqual(chain_a.generate(length=6), chain_b.generate(length=6))
    def test_stats_reports_expected_keys(self):
        chain = MarkovChain(order=2, seed=1)
        chain.train(["a", "b", "c", "a", "b", "d"])
        stats = chain.stats()
        for key in ("order", "unique_states", "total_transitions", "possible_start_points"):
            self.assertIn(key, stats)
class TestWordMarkovModel(unittest.TestCase):
    SAMPLE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "The lazy dog sleeps under the warm sun. "
        "A quick fox never sleeps during the day."
    )
    def test_train_produces_states(self):
        model = WordMarkovModel(order=2, seed=1)
        model.train(self.SAMPLE_TEXT)
        stats = model.stats()
        self.assertGreater(stats["unique_states"], 0)
    def test_generate_sentence_returns_string(self):
        model = WordMarkovModel(order=2, seed=1)
        model.train(self.SAMPLE_TEXT)
        sentence = model.generate_sentence(max_words=10)
        self.assertIsInstance(sentence, str)
        self.assertGreater(len(sentence.split()), 0)
    def test_generate_text_multiple_sentences(self):
        model = WordMarkovModel(order=1, seed=3)
        model.train(self.SAMPLE_TEXT)
        text = model.generate_text(num_sentences=3, max_words_per_sentence=10)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
    def test_empty_text_does_not_crash(self):
        model = WordMarkovModel(order=2)
        model.train("")
        self.assertEqual(model.stats()["unique_states"], 0)
class TestCharMarkovModel(unittest.TestCase):
    SAMPLE_TEXT = "alice alison alexis alina alistair olivia oliver ophelia orlando"
    def test_train_produces_states(self):
        model = CharMarkovModel(order=2, seed=1)
        model.train(self.SAMPLE_TEXT)
        stats = model.stats()
        self.assertGreater(stats["unique_states"], 0)
    def test_generate_word_returns_string(self):
        model = CharMarkovModel(order=2, seed=1)
        model.train(self.SAMPLE_TEXT)
        word = model.generate_word(max_length=12)
        self.assertIsInstance(word, str)
        self.assertGreater(len(word), 0)
    def test_generate_words_returns_correct_count(self):
        model = CharMarkovModel(order=2, seed=5)
        model.train(self.SAMPLE_TEXT)
        words = model.generate_words(count=5, max_length=10)
        self.assertEqual(len(words), 5)
        for w in words:
            self.assertIsInstance(w, str)
    def test_short_words_are_skipped_during_training(self):
        # order=3 means words with length <= 3 shouldn't contribute states
        model = CharMarkovModel(order=3, seed=1)
        model.train("a an the cat")
        # only "the" and "cat" are longer than order=3? "the"/"cat" are len 3,
        # so with order=3 they're skipped too -- expect zero states.
        self.assertEqual(model.stats()["unique_states"], 0)
if __name__ == "__main__":
    unittest.main()

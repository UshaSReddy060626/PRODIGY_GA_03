import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markov_gen import WordMarkovModel, CharMarkovModel
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_texts")
def word_level_demo():
    print("=" * 60)
    print("WORD-LEVEL MARKOV CHAIN -- sentence generation")
    print("=" * 60)
    with open(os.path.join(SAMPLE_DIR, "nature_essay.txt"), encoding="utf-8") as f:
        text = f.read()
    for order in (1, 2, 3):
        model = WordMarkovModel(order=order, seed=123)
        model.train(text)
        print(f"\n--- order={order} ---")
        print(model.generate_text(num_sentences=2, max_words_per_sentence=25))
def char_level_demo():
    print("\n" + "=" * 60)
    print("CHARACTER-LEVEL MARKOV CHAIN -- invented word generation")
    print("=" * 60)
    with open(os.path.join(SAMPLE_DIR, "fantasy_names.txt"), encoding="utf-8") as f:
        text = f.read()
    for order in (2, 3, 4):
        model = CharMarkovModel(order=order, seed=123)
        model.train(text)
        words = model.generate_words(count=8, max_length=14)
        print(f"\n--- order={order} ---")
        print(", ".join(w.capitalize() for w in words))
if __name__ == "__main__":
    word_level_demo()
    char_level_demo()

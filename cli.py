import argparse
import sys
from markov_gen import WordMarkovModel, CharMarkovModel
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text using a Markov chain trained on an input corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--file", "-f", type=str, default=None,
        help="Path to a UTF-8 text file to train on. If omitted, reads from stdin.",)
    parser.add_argument("--mode", "-m", choices=["word", "char"], default="word",
        help="'word' generates sentences; 'char' generates invented words.",)
    parser.add_argument("--order", "-o", type=int, default=None,
        help="Markov chain order (context length). Defaults: word=2, char=3.",)
    parser.add_argument("--sentences", type=int, default=3,
        help="[word mode] Number of sentences to generate.",)
    parser.add_argument("--max-words", type=int, default=30,
        help="[word mode] Max words per generated sentence.",)
    parser.add_argument("--words", type=int, default=10,
        help="[char mode] Number of invented words to generate.",)
    parser.add_argument("--max-length", type=int, default=20,
        help="[char mode] Max characters per generated word.",)
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed, for reproducible output.",)
    parser.add_argument("--stats", action="store_true",
        help="Print model statistics after training.",)
    return parser
def read_input(path: str) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if sys.stdin.isatty():
        print("No --file given and no piped input detected. Provide a text file or pipe text in.", file=sys.stderr)
        sys.exit(1)
    return sys.stdin.read()
def main() -> None:
    args = build_parser().parse_args()
    text = read_input(args.file)
    if not text.strip():
        print("Input text is empty -- nothing to train on.", file=sys.stderr)
        sys.exit(1)
    if args.mode == "word":
        order = args.order if args.order is not None else 2
        model = WordMarkovModel(order=order, seed=args.seed)
        model.train(text)
        if args.stats:
            print(f"[stats] {model.stats()}", file=sys.stderr)
        for i in range(args.sentences):
            print(model.generate_sentence(max_words=args.max_words))
    else: 
        order = args.order if args.order is not None else 3
        model = CharMarkovModel(order=order, seed=args.seed)
        model.train(text)
        if args.stats:
            print(f"[stats] {model.stats()}", file=sys.stderr)
        words = model.generate_words(count=args.words, max_length=args.max_length)
        print(", ".join(words))
if __name__ == "__main__":
    main()

# 📝 Text Generation with Markov Chains

> Generate new text and creative words by learning statistical patterns from an existing text corpus.

---

## 📌 Overview

This project implements a **Markov Chain-based text generator from scratch using pure Python**. No external machine learning libraries are required.

The model learns which tokens are likely to appear after a fixed sequence of previous tokens and uses the learned transition frequencies to generate new text that statistically resembles the training corpus.

Two approaches are implemented:

| Model             | Token Unit | Purpose                           |
| ----------------- | ---------- | --------------------------------- |
| `WordMarkovModel` | Words      | Generate new sentences            |
| `CharMarkovModel` | Characters | Generate invented words and names |

Both models use the same reusable **generic `MarkovChain` engine**, allowing the core transition and sampling logic to be implemented and tested only once.

---

# 🧠 What is a Markov Chain?

A **Markov Chain** is a probabilistic model in which the next state depends only on the current state rather than the complete history.

For text generation, this means that the next token depends on a fixed number of previous tokens.

For example, with:

```text
order = 2
```

the model might learn:

```text
"The cat" → "sat"
"The cat" → "ran"
"The cat" → "slept"
```

If `"sat"` appears more frequently after `"The cat"` than the other words, it will have a higher probability of being selected.

The basic process is:

```text
Previous Tokens
       ↓
Transition Probabilities
       ↓
    Next Token
```

---

# 🔄 How the Text Generator Works

The complete text-generation process consists of three major stages:

## 1️⃣ Tokenization

The input corpus is converted into a sequence of tokens.

### 📝 Word-Level Model

Input:

```text
"The sun rises above the hills"
```

becomes:

```text
["The", "sun", "rises", "above", "the", "hills"]
```

### 🔤 Character-Level Model

Input:

```text
"Arwen"
```

becomes:

```text
["A", "r", "w", "e", "n"]
```

---

## 2️⃣ Build the Transition Table

For every sequence of `order` tokens, the model records what token comes next.

For example, with:

```text
order = 2
```

the model may learn:

```text
("the", "cat") → ["sat", "ran", "sat"]
```

The frequency information becomes:

```text
"sat" → 2 occurrences
"ran" → 1 occurrence
```

Therefore:

```text
P(sat | the, cat) = 2/3
P(ran | the, cat) = 1/3
```

The model samples from these learned frequencies rather than selecting every possible token with equal probability.

---

## 3️⃣ Generate New Text

Generation begins from a state observed in the training data.

The model then follows this process:

```text
Choose Starting State
        ↓
Find Possible Next Tokens
        ↓
Sample Using Learned Frequencies
        ↓
Add Selected Token
        ↓
Move Context Window Forward
        ↓
Repeat
```

This continues until the desired number of words or characters has been generated.

---

# 📐 The Markov Property

The central assumption of a Markov model is:

```text
P(next token | entire history)
             ↓
P(next token | recent N tokens)
```

For an order-2 model:

```text
P("sat" | "the", "cat")
```

depends only on:

```text
"the", "cat"
```

rather than every word that appeared before them.

This makes the model:

* ⚡ Simple
* 🚀 Fast
* 🧩 Easy to implement
* 🧪 Easy to experiment with

---

# 🔤 Two Types of Models

## 📝 Word Markov Model

The `WordMarkovModel` operates at the **word level**.

```text
Input:
"The moon shines above the quiet forest."

        ↓

Word Tokens:
The → moon → shines → above → the → quiet → forest

        ↓

Markov Chain

        ↓

Generated Text
```

### 🎯 Best For

* Sentence generation
* Creative writing experiments
* Learning language patterns
* Demonstrating n-gram models

---

## 🔡 Character Markov Model

The `CharMarkovModel` operates at the **character level**.

For example:

```text
"Elowen"
```

becomes:

```text
E → l → o → w → e → n
```

The model learns character-level patterns and combines them to create new words.

### 🎯 Best For

* Fantasy names
* Character names
* Brand-name ideas
* Random word generation
* Creative experiments

---

# 🎛️ The `order` Parameter

The `order` parameter determines how much recent context the model considers.

## 🔹 Lower Order

```text
order = 1
```

The model considers only one previous token:

```text
Previous Word → Next Word
```

This generally produces:

* 🎲 More variation
* 🌪️ More randomness
* 📉 Less grammatical consistency

---

## 🔸 Medium Order

```text
order = 2
```

The model considers two previous tokens:

```text
Previous 2 Words → Next Word
```

This often provides a balance between:

```text
Novelty ↔ Coherence
```

---

## 🔺 Higher Order

```text
order = 3
```

The model considers three previous tokens.

This generally produces text that is:

* 🧠 More coherent
* 📚 More similar to the training corpus
* 🎲 Less novel

As the order increases sufficiently, the model may begin reproducing phrases from the original training data.

---

# 📊 Order vs Output

The general trade-off can be summarized as:

### Lower Order

```text
Lower Order
     ↓
More Novelty
     ↓
Less Context
     ↓
More Random Output
```

### Higher Order

```text
Higher Order
     ↓
More Context
     ↓
Higher Coherence
     ↓
Closer to Training Data
```

Therefore, choosing the appropriate order depends on the desired output.

---

# 🏗️ Project Architecture

```text
                    Training Corpus
                          ↓
                     Tokenization
                          ↓
                  ┌─────────────────┐
                  │   MarkovChain   │
                  │  Generic Engine │
                  └─────────────────┘
                     ↙           ↘
                   ↙               ↘
          WordMarkovModel     CharMarkovModel
                ↓                   ↓
       Sentence Generation    Word Generation
```

The reusable `MarkovChain` class works with any sequence of **hashable tokens**, allowing both word-level and character-level models to share the same underlying implementation.

---

# 📁 Project Structure

```text
markov-text-generator/
│
├── markov_gen/
│   ├── __init__.py
│   ├── markov_chain.py
│   ├── word_model.py
│   └── char_model.py
│
├── cli.py
│
├── examples/
│   └── demo.py
│
├── sample_texts/
│   ├── nature_essay.txt
│   └── fantasy_names.txt
│
├── tests/
│   └── test_markov.py
│
├── requirements.txt
└── README.md
```

## 📄 File Overview

| File               | Purpose                                       |
| ------------------ | --------------------------------------------- |
| `markov_chain.py`  | Generic order-N Markov chain engine           |
| `word_model.py`    | Word-level sentence generation                |
| `char_model.py`    | Character-level word generation               |
| `cli.py`           | Command-line interface                        |
| `demo.py`          | Demonstrates both models and different orders |
| `test_markov.py`   | Unit tests                                    |
| `sample_texts/`    | Example training corpora                      |
| `requirements.txt` | Optional testing dependency                   |

---

# ⚙️ Installation

No external machine learning libraries are required.

The core implementation uses only the **Python Standard Library**.

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/markov-text-generator.git
cd markov-text-generator
```

## 2️⃣ Install Testing Dependencies

If you want to run the test suite using `pytest`:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1️⃣ Command-Line Interface

### 📝 Generate Sentences

```bash
python cli.py --file sample_texts/nature_essay.txt --mode word --order 2 --sentences 5
```

### 🔤 Generate Invented Words

```bash
python cli.py --file sample_texts/fantasy_names.txt --mode char --order 3 --words 10
```

### 📥 Use Standard Input

Text can also be piped directly into the program:

```bash
cat my_notes.txt | python cli.py --mode word --sentences 3
```

### ❓ View All Options

```bash
python cli.py --help
```

---

# 🎛️ Command-Line Options

| Flag            | Description                                | Default                |
| --------------- | ------------------------------------------ | ---------------------- |
| `--file`, `-f`  | Training text file; reads stdin if omitted | —                      |
| `--mode`, `-m`  | `word` or `char`                           | `word`                 |
| `--order`, `-o` | Context/state size                         | 2 for word, 3 for char |
| `--sentences`   | Number of sentences in word mode           | 3                      |
| `--max-words`   | Maximum words per sentence                 | 30                     |
| `--words`       | Number of generated words in char mode     | 10                     |
| `--max-length`  | Maximum characters per generated word      | 20                     |
| `--seed`        | Random seed for reproducible output        | Random                 |
| `--stats`       | Display model statistics                   | Off                    |

---

# 🐍 Using the Library

The models can also be imported directly into Python programs.

```python
from markov_gen import WordMarkovModel, CharMarkovModel

model = WordMarkovModel(order=2, seed=42)

model.train(
    open("sample_texts/nature_essay.txt").read()
)

print(model.generate_text(num_sentences=3))


name_model = CharMarkovModel(order=3, seed=42)

name_model.train(
    open("sample_texts/fantasy_names.txt").read()
)

print(name_model.generate_words(count=10))
```

---

# 🧪 Example Script

Run the bundled demonstration:

```bash
python examples/demo.py
```

The demo:

* 📝 Trains the word-level model
* 🔤 Trains the character-level model
* 🎛️ Experiments with different `order` values
* 📤 Displays generated outputs
* 🧠 Demonstrates the relationship between context size and creativity

---

# 📤 Sample Output

## 📝 Word-Level Generation

Training corpus:

```text
sample_texts/nature_essay.txt
```

Example:

```text
--- order=1 ---

Mist curls between the forest wakes slowly past its peak, changing shape
as the last light turns soft and somehow more honest beneath the treetops...

--- order=2 ---

Mist curls between the tall pines like smoke from a quiet fire.

The wind carries them onward.

--- order=3 ---

Mist curls between the tall pines like smoke from a quiet fire.
```

The output demonstrates how changing the Markov order affects the relationship between randomness and coherence.

---

## 🔤 Character-Level Generation

Training corpus:

```text
sample_texts/fantasy_names.txt
```

Example:

```text
--- order=2 ---

Gidelphyris, Ulrintara, Lyssiambrick, Quen, Zin, Kaelyssaellith

--- order=3 ---

Gideon, Nerick, Quirin, Ravenne, Lyssa, Orvath

--- order=4 ---

Gideon, Idris, Winterhold, Fiora, Faelan, Aldrin
```

The results demonstrate how increasing the order generally causes generated words to become more similar to patterns found in the training corpus.

---

# 🎲 Reproducible Generation

Each model accepts an optional random seed.

For example:

```python
model = WordMarkovModel(order=2, seed=42)
```

Using the same:

```text
Training Data
      +
Model Configuration
      +
Random Seed
```

allows the generated output to be reproduced consistently.

This is particularly useful for:

* 🧪 Testing
* 🐛 Debugging
* 🎓 Demonstrations
* 💼 Internship submissions
* 📊 Comparing model configurations

---

# 🛡️ Edge Cases Handled

The implementation deliberately handles several practical situations.

## 🚫 Dead-End States

If generation reaches a state with no known continuation, the model stops gracefully instead of crashing.

## 📝 Sentence-Aware Training

The word model trains one sentence at a time so generation can begin from valid sentence beginnings.

## 🔚 End-of-Word Handling

The character model uses an explicit end-of-word marker so generated words can terminate naturally rather than continuing indefinitely.

## 🎲 Randomness Control

Optional random seeds make generated results reproducible.

---

# 🧪 Running Tests

## Using Python's Built-in `unittest`

```bash
python -m unittest discover tests -v
```

## Using `pytest`

```bash
pytest tests/ -v
```

The tests verify the core Markov Chain behavior and help ensure that both word-level and character-level models work correctly.

---

# 🧩 Design Highlights

## ♻️ Reusable Markov Engine

Instead of implementing separate transition logic for words and characters, both models use:

```text
MarkovChain
```

as their underlying engine.

This keeps the implementation:

* 🧩 Modular
* ♻️ Reusable
* 🧪 Easier to test
* 🚀 Easier to extend

---

## 🔤 Token-Agnostic Design

The generic engine can operate on any sequence of **hashable tokens**.

Therefore, the same core logic can be applied to:

```text
Words
Characters
Numbers
Symbols
Other discrete tokens
```

---

## 📈 Frequency-Based Sampling

The model preserves the frequency of observed transitions.

For example:

```text
("the", "cat") → ["sat", "sat", "ran"]
```

results in:

```text
P(sat) = 2/3
P(ran) = 1/3
```

This produces more natural statistical behavior than uniformly selecting a possible next token.

---

# 💡 What This Project Demonstrates

This project provides practical experience with:

* 🔗 Markov Chains
* 📐 Markov Property
* 🔢 N-gram Models
* 🎲 Probabilistic Text Generation
* 🔤 Tokenization
* 🔄 Transition Probabilities
* 📊 Frequency-Based Sampling
* 🎲 Random Sampling
* 🧠 Natural Language Processing Fundamentals
* 🔡 Character-Level Language Modeling
* 📝 Word-Level Language Modeling
* 🐍 Python OOP
* 🧩 Modular Software Design
* 🧪 Unit Testing
* 💻 Command-Line Interfaces

---

# ⚠️ Limitations

Markov chains are useful for understanding the foundations of probabilistic language generation, but they have important limitations.

## 📉 Limited Context

The model only considers a fixed number of previous tokens.

## 🧠 No Semantic Understanding

The model learns statistical patterns rather than meaning.

## 🔁 Repetition

Higher-order models can reproduce phrases from the training corpus.

## 📝 Coherence

Longer generated text may become grammatically or semantically inconsistent.

## 📚 Data Dependency

The quality and diversity of generated text depend heavily on the training corpus.

These limitations highlight why modern language models use substantially more sophisticated architectures and longer-range context mechanisms.

---

# 🔮 Future Improvements

Possible extensions include:

* [ ] Add text preprocessing and punctuation handling
* [ ] Add configurable start/end tokens
* [ ] Add transition probability visualization
* [ ] Add perplexity-style evaluation
* [ ] Add a graphical interface
* [ ] Add a Streamlit demo
* [ ] Support larger text corpora
* [ ] Add model serialization and loading
* [ ] Add more advanced tokenization
* [ ] Compare different Markov orders automatically
* [ ] Add corpus statistics and vocabulary analysis
* [ ] Compare Markov generation with an RNN/LSTM language model
* [ ] Compare generated text with a Transformer-based language model

---

# 📚 References

* [Text Generation with Markov Chains — Towards Data Science](https://towardsdatascience.com/text-generation-with-markov-chains-an-introduction-to-using-markovify-742e6680dc33)
* [Allison Parrish — Predictive Text and Text Generation](https://github.com/aparrish/predictive-text-and-text-generation)

---

# ⭐ Conclusion

This project implements a **Markov Chain text generator completely from scratch using Python's Standard Library**.

It demonstrates how simple statistical models can generate new text by learning transition patterns from a corpus.

The project implements both:

```text
Word-Level Markov Model
        ↓
Sentence Generation
```

and:

```text
Character-Level Markov Model
        ↓
Invented Word Generation
```

The most important concept demonstrated by the project is the relationship between **context size, randomness, and coherence**.

```text
Lower Order
     ↓
More Novelty

Higher Order
     ↓
More Coherence
     ↓
Closer to Training Data
```

Although Markov chains are much simpler than modern language models, they provide an excellent foundation for understanding the probabilistic ideas behind text generation.

---

# 👩‍💻 Author

**Usha S. Reddy**

If you found this project useful, consider giving the repository a ⭐.

import random
from collections import defaultdict
from typing import Hashable, List, Sequence, Tuple, Dict
class MarkovChain:
    """A generic order-N Markov chain over a sequence of tokens."""
    def __init__(self, order: int = 2, seed: int = None):
        """
        Args:
            order: How many previous tokens form a "state". Higher order
                   means output stays closer to the source text (less
                   creative, more coherent); lower order means more
                   random/novel output.
            seed:  Optional random seed for reproducible generation.
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # state (tuple of `order` tokens) -> list of tokens seen next
        # (kept as a list, not a set, so frequency/weighting is preserved)
        self.transitions: Dict[Tuple[Hashable, ...], List[Hashable]] = defaultdict(list)
        # states that are valid starting points for generation
        self.start_states: List[Tuple[Hashable, ...]] = []
        self._rng = random.Random(seed)
    def train(self, tokens: Sequence[Hashable], is_start_of_sequence: bool = True) -> None:
        """
        Feed one sequence of tokens into the model, updating the
        transition table.
        Args:
            tokens: The sequence (e.g. words in a sentence, or characters
                    in a word) to learn from.
            is_start_of_sequence: If True, the very first state of this
                    sequence is recorded as a valid generation starting
                    point. Set to False when concatenating chunks that
                    aren't natural sentence/word starts.
        """
        tokens = list(tokens)
        if len(tokens) <= self.order:
            return
        if is_start_of_sequence:
            self.start_states.append(tuple(tokens[: self.order]))
        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i : i + self.order])
            next_token = tokens[i + self.order]
            self.transitions[state].append(next_token)
    def generate(self, length: int = 50, start_state: Tuple[Hashable, ...] = None) -> List[Hashable]:
        """
        Generate a sequence of tokens from the trained model.

        Args:
            length: Maximum number of tokens to generate (generation may
                    stop earlier if a dead-end state with no known
                    transitions is reached).
            start_state: Optional explicit starting state (tuple of
                    `order` tokens). Must be a state the model has seen.
                    If omitted, a random valid start state is chosen.

        Returns:
            List of generated tokens (including the seed/start tokens).
        """
        if not self.transitions:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        if start_state is None:
            if not self.start_states:
                start_state = self._rng.choice(list(self.transitions.keys()))
            else:
                start_state = self._rng.choice(self.start_states)
        if start_state not in self.transitions:
            raise ValueError(f"Unseen start state: {start_state!r}")
        result = list(start_state)
        state = start_state
        while len(result) < length:
            choices = self.transitions.get(state)
            if not choices:
                break  # dead end -- no known continuation for this state
            next_token = self._rng.choice(choices)
            result.append(next_token)
            state = tuple(result[-self.order :])
        return result
    def stats(self) -> dict:
        """Return basic info about the trained model (useful for debugging/README demos)."""
        return {
            "order": self.order,
            "unique_states": len(self.transitions),
            "total_transitions": sum(len(v) for v in self.transitions.values()),
            "possible_start_points": len(self.start_states),}

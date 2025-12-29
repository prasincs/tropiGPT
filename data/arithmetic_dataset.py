"""
ArithmeticDataset: Character-level arithmetic data pipeline with Abacus Embeddings.

Implements the "Abacus Embeddings" approach where digits at the same significance
(same power of 10) receive the same positional embedding, helping transformers
learn to align carries properly.

Reference: "Abacus Embeddings give the same positional embeddings to all digits
of the same significance."
"""

import random
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader


# Vocabulary: character-level tokenization (no BPE)
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', ' ', '\n']
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# Special tokens
PAD_ID = CHAR_TO_ID[' ']
PLUS_ID = CHAR_TO_ID['+']
EQUALS_ID = CHAR_TO_ID['=']
NEWLINE_ID = CHAR_TO_ID['\n']


class ArithmeticDataset(IterableDataset):
    """
    Infinite PyTorch IterableDataset for addition problems.

    Generates problems of the form: "a + b = c" with character-level tokenization
    and Abacus-style significance embeddings.

    Args:
        min_digits: Minimum number of digits for operands (default: 1)
        max_digits: Maximum number of digits for operands (default: 3)
        reverse_digits: If True, reverse digit order (LSB first) for better carry learning
        seed: Random seed for reproducibility (None for random)
        seq_length: Fixed sequence length (pads/truncates to this length)

    Returns per sample:
        input_ids: Token IDs for the problem string
        significance_ids: Abacus embeddings (0 for non-digits, 1-N for digit significance)
        target_ids: Next-token prediction targets (input_ids shifted by 1)
    """

    def __init__(
        self,
        min_digits: int = 1,
        max_digits: int = 3,
        reverse_digits: bool = False,
        seed: Optional[int] = None,
        seq_length: Optional[int] = None,
    ):
        super().__init__()
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.reverse_digits = reverse_digits
        self.seed = seed
        self.seq_length = seq_length

        # Calculate max sequence length if not provided
        # Format: "a + b = c\n" where c can have max_digits + 1 digits (carry)
        # Max length: max_digits + 3 + max_digits + 3 + (max_digits + 1) + 1
        if self.seq_length is None:
            self.seq_length = 3 * self.max_digits + 8  # Conservative estimate

    def _generate_problem(self, rng: random.Random) -> Tuple[str, str, str]:
        """Generate a single addition problem: (a_str, b_str, result_str)."""
        num_digits_a = rng.randint(self.min_digits, self.max_digits)
        num_digits_b = rng.randint(self.min_digits, self.max_digits)

        # Generate random numbers with the specified digit counts
        min_a = 10 ** (num_digits_a - 1) if num_digits_a > 1 else 0
        max_a = 10 ** num_digits_a - 1
        min_b = 10 ** (num_digits_b - 1) if num_digits_b > 1 else 0
        max_b = 10 ** num_digits_b - 1

        a = rng.randint(min_a, max_a)
        b = rng.randint(min_b, max_b)
        c = a + b

        return str(a), str(b), str(c)

    def _compute_significance(self, num_str: str) -> list:
        """
        Compute significance IDs for a number string.

        Significance = power of 10 + 1 (so units = 1, tens = 2, etc.)
        If reversed, the string is already LSB-first, so index 0 is units.
        If not reversed, we need to compute from the right.

        Args:
            num_str: String representation of the number (possibly reversed)

        Returns:
            List of significance values (1 for 10^0, 2 for 10^1, etc.)
        """
        n = len(num_str)
        if self.reverse_digits:
            # String is LSB-first: index 0 is units (10^0), index 1 is tens (10^1)
            return [i + 1 for i in range(n)]
        else:
            # String is MSB-first: rightmost is units (10^0)
            return [n - i for i in range(n)]

    def _format_problem(self, a_str: str, b_str: str, c_str: str) -> str:
        """Format the problem as a string, optionally reversing digits."""
        if self.reverse_digits:
            a_str = a_str[::-1]
            b_str = b_str[::-1]
            c_str = c_str[::-1]

        return f"{a_str} + {b_str} = {c_str}\n"

    def _tokenize_with_significance(self, problem: str, a_str: str, b_str: str, c_str: str) -> Tuple[list, list]:
        """
        Tokenize the problem string and compute significance IDs.

        Args:
            problem: Full problem string (e.g., "123 + 456 = 579\n")
            a_str, b_str, c_str: Original number strings (before any reversal)

        Returns:
            (token_ids, significance_ids)
        """
        # Apply reversal if needed (for significance computation)
        if self.reverse_digits:
            a_str = a_str[::-1]
            b_str = b_str[::-1]
            c_str = c_str[::-1]

        token_ids = []
        significance_ids = []

        # Parse the problem string and assign significance
        # Format: "{a} + {b} = {c}\n"
        idx = 0

        # Parse first number (a)
        for i, ch in enumerate(a_str):
            token_ids.append(CHAR_TO_ID[ch])
            significance_ids.append(self._compute_significance(a_str)[i])
            idx += 1

        # Parse " + "
        for ch in " + ":
            token_ids.append(CHAR_TO_ID[ch])
            significance_ids.append(0)  # Non-digit
            idx += 1

        # Parse second number (b)
        for i, ch in enumerate(b_str):
            token_ids.append(CHAR_TO_ID[ch])
            significance_ids.append(self._compute_significance(b_str)[i])
            idx += 1

        # Parse " = "
        for ch in " = ":
            token_ids.append(CHAR_TO_ID[ch])
            significance_ids.append(0)  # Non-digit
            idx += 1

        # Parse result (c)
        for i, ch in enumerate(c_str):
            token_ids.append(CHAR_TO_ID[ch])
            significance_ids.append(self._compute_significance(c_str)[i])
            idx += 1

        # Newline
        token_ids.append(NEWLINE_ID)
        significance_ids.append(0)

        return token_ids, significance_ids

    def _pad_or_truncate(self, ids: list, pad_value: int = 0) -> list:
        """Pad or truncate a list to self.seq_length."""
        if len(ids) >= self.seq_length:
            return ids[:self.seq_length]
        return ids + [pad_value] * (self.seq_length - len(ids))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield infinite stream of (input_ids, significance_ids, target_ids)."""
        # Handle worker processes for DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Different seed per worker
            worker_seed = self.seed + worker_info.id if self.seed else random.randint(0, 2**32 - 1)
            rng = random.Random(worker_seed)
        else:
            rng = random.Random(self.seed)

        while True:
            # Generate problem
            a_str, b_str, c_str = self._generate_problem(rng)
            problem = self._format_problem(a_str, b_str, c_str)

            # Tokenize with significance
            token_ids, significance_ids = self._tokenize_with_significance(
                problem, a_str, b_str, c_str
            )

            # Create target (next token prediction)
            target_ids = token_ids[1:] + [PAD_ID]

            # Pad/truncate to fixed length
            input_ids = self._pad_or_truncate(token_ids, PAD_ID)
            significance_ids = self._pad_or_truncate(significance_ids, 0)
            target_ids = self._pad_or_truncate(target_ids, PAD_ID)

            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(significance_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long),
            )


def decode_tokens(token_ids: torch.Tensor) -> str:
    """Decode token IDs back to string."""
    return ''.join(ID_TO_CHAR[i.item()] for i in token_ids if i.item() in ID_TO_CHAR)


def test_batch():
    """
    Test the ArithmeticDataset and print aligned samples to verify logic.
    """
    print("=" * 70)
    print("ArithmeticDataset Test")
    print("=" * 70)

    # Test 1: Standard order (MSB first)
    print("\n[Test 1] Standard order (MSB first), 1-3 digits:")
    print("-" * 50)

    dataset = ArithmeticDataset(
        min_digits=1,
        max_digits=3,
        reverse_digits=False,
        seed=42,
        seq_length=20,
    )

    loader = DataLoader(dataset, batch_size=3)
    batch = next(iter(loader))
    input_ids, significance_ids, target_ids = batch

    for i in range(len(input_ids)):
        tokens = decode_tokens(input_ids[i])
        print(f"Problem:      '{tokens.strip()}'")
        print(f"Chars:        {[c for c in tokens]}")
        print(f"Token IDs:    {input_ids[i].tolist()}")
        print(f"Significance: {significance_ids[i].tolist()}")
        print(f"Target IDs:   {target_ids[i].tolist()}")
        print()

    # Test 2: Reversed order (LSB first)
    print("\n[Test 2] Reversed order (LSB first), 1-3 digits:")
    print("-" * 50)

    dataset_rev = ArithmeticDataset(
        min_digits=1,
        max_digits=3,
        reverse_digits=True,
        seed=42,
        seq_length=20,
    )

    loader_rev = DataLoader(dataset_rev, batch_size=3)
    batch_rev = next(iter(loader_rev))
    input_ids_rev, significance_ids_rev, target_ids_rev = batch_rev

    for i in range(len(input_ids_rev)):
        tokens = decode_tokens(input_ids_rev[i])
        print(f"Problem:      '{tokens.strip()}'")
        print(f"Chars:        {[c for c in tokens]}")
        print(f"Token IDs:    {input_ids_rev[i].tolist()}")
        print(f"Significance: {significance_ids_rev[i].tolist()}")
        print(f"Target IDs:   {target_ids_rev[i].tolist()}")
        print()

    # Test 3: Specific example '99 + 1'
    print("\n[Test 3] Manual verification with '99 + 1':")
    print("-" * 50)

    # Create a simple dataset and manually check one example
    dataset_manual = ArithmeticDataset(
        min_digits=1,
        max_digits=2,
        reverse_digits=False,
        seed=None,
        seq_length=15,
    )

    # Manually create the '99 + 1 = 100' example
    a_str, b_str, c_str = "99", "1", "100"
    problem = f"{a_str} + {b_str} = {c_str}\n"
    token_ids, significance_ids = dataset_manual._tokenize_with_significance(
        problem, a_str, b_str, c_str
    )

    print(f"Problem: '99 + 1 = 100'")
    print()
    print("Aligned view:")
    print(f"Char:         {'  '.join(list(problem.strip()))}")
    print(f"Token ID:     {'  '.join(str(t) for t in token_ids[:-1])}")  # Skip newline for alignment
    print(f"Significance: {'  '.join(str(s) for s in significance_ids[:-1])}")
    print()
    print("Explanation:")
    print("  - '9' (tens place):      significance = 2 (10^1)")
    print("  - '9' (units place):     significance = 1 (10^0)")
    print("  - ' ', '+', '=':         significance = 0 (non-digit)")
    print("  - '1' in '1':            significance = 1 (10^0)")
    print("  - '1' in '100':          significance = 3 (10^2)")
    print("  - '0' in '100' (tens):   significance = 2 (10^1)")
    print("  - '0' in '100' (units):  significance = 1 (10^0)")

    # Test 4: Reversed version of '99 + 1'
    print("\n[Test 4] Reversed '99 + 1' (LSB first):")
    print("-" * 50)

    dataset_rev_manual = ArithmeticDataset(
        min_digits=1,
        max_digits=2,
        reverse_digits=True,
        seed=None,
        seq_length=15,
    )

    problem_rev = f"{'99'[::-1]} + {'1'[::-1]} = {'100'[::-1]}\n"
    token_ids_rev, significance_ids_rev = dataset_rev_manual._tokenize_with_significance(
        problem_rev, "99", "1", "100"
    )

    print(f"Problem (reversed): '99 + 1 = 001'")
    print()
    print("Aligned view:")
    chars_rev = list(problem_rev.strip())
    print(f"Char:         {'  '.join(chars_rev)}")
    print(f"Token ID:     {'  '.join(str(t) for t in token_ids_rev[:-1])}")
    print(f"Significance: {'  '.join(str(s) for s in significance_ids_rev[:-1])}")
    print()
    print("Explanation (reversed - LSB first):")
    print("  - First '9':   significance = 1 (units, 10^0)")
    print("  - Second '9':  significance = 2 (tens, 10^1)")
    print("  - '1':         significance = 1 (units, 10^0)")
    print("  - '0':         significance = 1 (units, 10^0)")
    print("  - '0':         significance = 2 (tens, 10^1)")
    print("  - '1':         significance = 3 (hundreds, 10^2)")

    print("\n" + "=" * 70)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Vocabulary: {CHARS}")
    print("=" * 70)


if __name__ == "__main__":
    test_batch()

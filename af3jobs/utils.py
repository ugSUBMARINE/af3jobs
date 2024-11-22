"""module for various utility functions"""

from itertools import product


def chain_id(letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Generator function to yield mmCIF chain IDs.

    This generator sequentially produces unique chain identifiers commonly used in mmCIF files.
    It generates all uppercase single-letter IDs ('A', 'B', ..., 'Z') followed by all possible
    two-letter combinations in 'reverse spreadsheet style' ('AA', 'BA', ..., 'ZZ').

    Yields:
        str: A unique chain ID in the sequence described.
    """
    # Yield single uppercase letters first
    yield from letters

    # Yield combinations of two uppercase letters
    for combination in product(letters, repeat=2):
        yield "".join(reversed(combination))

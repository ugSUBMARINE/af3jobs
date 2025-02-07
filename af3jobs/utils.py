"""module for various utility functions and classes"""

from __future__ import annotations

import json
from itertools import product
from typing import Generator


def chain_id(letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> Generator[str, None, None]:
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


def get_msa_from_json(json_file: str, sequence: str, paired: bool = False) -> str | None:
    """
    Read a JSON file generated by AF3 (<job_name>_data.json) and return the
    multiple sequence alignment (MSA) for the protein/RNA chain with
    the specified sequence.
    Default is to return the unpaired MSA, but the paired MSA can be returned for proteins.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    for entry in data["sequences"]:
        if chain := entry.get("protein"):
            if chain["sequence"] == sequence:
                return chain["unpairedMsa"] if not paired else chain["pairedMsa"]
        if chain := entry.get("rna"):
            if chain["sequence"] == sequence:
                return chain["unpairedMsa"]


def get_msa_from_a3m(a3m_file: str, sequence: str) -> str | None:
    """
    Read a multiple sequence alignment (MSA) file in A3M format and return the
    MSA as a string if the query sequence matches the specified sequence.
    """
    # get the first sequence in the A3M file
    seq_gen = fasta_sequences(a3m_file)
    _, query_sequence = next(seq_gen)
    if query_sequence.upper().replace("-", "") == sequence:
        with open(a3m_file, "r") as f:
            msa = f.read()
            return msa


def get_templates_from_json(json_file: str, sequence: str) -> list | None:
    """
    Read a JSON file generated by AF3 (<job_name>_data.json) and return the
    list of templates used for modeling the protein chain with
    the specified sequence.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    for entry in data["sequences"]:
        if chain := entry.get("protein"):
            if chain["sequence"] == sequence:
                return chain["templates"]


def fasta_sequences(fasta_file: str) -> Generator[tuple[str, str], None, None]:
    """
    Read a FASTA formatted file and yield tuples with the title and the sequence
    """
    with open(fasta_file, "r") as f:
        title = ""
        sequence = ""
        for line in f:
            if line.startswith(">"):
                if title:
                    yield title, sequence
                title = line.strip()
                sequence = ""
            else:
                sequence += line.strip()
        yield title, sequence

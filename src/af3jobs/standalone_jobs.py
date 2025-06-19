"""
Module for defining molecular components, modifications, and job configurations for AlphaFold 3.

The `Job` class is the main container for combining chains, ligands/ions, and modifications that can be converted
to JSON as input for AlphaFold 3.

It follows the JSON schema defined in the AlphaFold 3 documentation (as of November 2024):
https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from itertools import islice
from random import randint
from typing import Any, Self, Type

from .components import DnaChain, Ligand, ProteinChain, RnaChain
from .utils import chain_id

# type definitions
Sequence = ProteinChain | DnaChain | RnaChain | Ligand
Atom = tuple[str, int, str]
Bond = tuple[Atom, Atom]


@dataclass
class Job:
    """Represents an AlphaFold3 job with methods to add entities."""

    name: str
    model_seeds: list[int] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)
    bonded_atom_pairs: list[Bond] = field(default_factory=list)
    user_ccd: str = ""
    dialect: str = "alphafold3"
    version: int = 2  # default version

    def __post_init__(self) -> None:
        """Post-initialization method."""
        if not self.model_seeds:
            # generate a random model seed
            self.model_seeds = [randint(1, 1 << 31)]

        # Initialize the chain ID generator
        self._chain_ids = chain_id()

    def __str__(self) -> str:
        """Return a string representation of the job."""
        lines = [
            f"Job name: {self.name}",
            f"Dialect: {self.dialect}",
            f"Version: {self.version}",
            f"Model seeds: {self.model_seeds}",
        ]
        num_seq = len(self.sequences)
        if num_seq == 0:
            lines.append("No sequences.")
        else:
            lines.append(f"{num_seq} Sequence(s):")
            for seq in self.sequences:
                lines.append(str(seq))

        if self.bonded_atom_pairs:
            lines.append("Bonded atom pairs:")
            for pair in self.bonded_atom_pairs:
                lines.append(f"  {pair[0]} -- {pair[1]}")

        if self.user_ccd:
            # get the names of the compounds defined in the user CCD
            data = [
                line.split("data_")[-1]
                for line in self.user_ccd.splitlines()
                if line.startswith("data_")
            ]
            lines.append(f"User CCD: {', '.join(data)}")

        return "\n".join(lines)

    def _check_ids(self) -> None:
        """Check for duplicate IDs."""
        seq_ids: list[str] = []
        for seq in self.sequences:
            seq_ids.extend(seq.ids)
        if len(seq_ids) != len(set(seq_ids)):
            raise ValueError(f"Duplicate chain IDs found: {seq_ids}")

    def _get_ids(self, ids: None | str | list[str], count: int) -> list[str]:
        if ids is None:
            ids = list(islice(self._chain_ids, count))
        else:
            if isinstance(ids, str):
                ids = [ids]
            elif not isinstance(ids, list):
                raise TypeError("IDs must be a string or a list of strings.")
        if not ids:
            raise ValueError("Number of chains or ligands must be greater than zero.")
        return ids

    def add_protein_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> ProteinChain:
        """Add a protein chain to the job."""
        chn = ProteinChain(self._get_ids(ids, count), sequence)
        self.sequences.append(chn)
        return chn

    def add_dna_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> DnaChain:
        """Add a DNA chain to the job."""
        chn = DnaChain(self._get_ids(ids, count), sequence)
        self.sequences.append(chn)
        return chn

    def add_rna_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> RnaChain:
        """Add an RNA chain to the job."""
        chn = RnaChain(self._get_ids(ids, count), sequence)
        self.sequences.append(chn)
        return chn

    def add_ligand(
        self,
        ccd_codes: str | list[str] | None = None,
        smiles: str = "",
        count: int = 1,
        ids: None | str | list[str] = None,
    ) -> Ligand:
        """Add a ligand to the job."""
        if not ccd_codes and not smiles:
            raise ValueError("Either CCD codes or SMILES string must be provided.")
        if ccd_codes and smiles:
            warnings.warn(
                "`ccd_codes` and `smiles` are given - they are mutually exclusive - will be using smiles"
            )
        _ids = self._get_ids(ids, count)
        if smiles:
            ligand = Ligand(_ids, smiles=smiles)
        else:
            if isinstance(ccd_codes, str):
                ccd_codes = [ccd_codes]
            ligand = Ligand(_ids, ccd_codes=ccd_codes)
        self.sequences.append(ligand)
        return ligand

    def add_bonded_atom_pair(
        self, id_1: str, resi_1: int, name_1: str, id_2: str, resi_2: int, name_2: str
    ) -> None:
        """Add a bonded atom pair to the job."""
        self.bonded_atom_pairs.append(((id_1, resi_1, name_1), (id_2, resi_2, name_2)))

    def to_dict(self) -> dict[str, Any]:
        """Convert the Job to a dictionary suitable for JSON serialization."""
        d: dict[str, Any] = {
            "name": self.name,
            "modelSeeds": self.model_seeds,
        }

        # add sequences / ligands / ions
        if self.sequences:
            self._check_ids()
            d["sequences"] = [seq.to_dict(self.version) for seq in self.sequences]
        else:
            raise ValueError("Empty list of sequences.")

        # add bonded atom pairs
        if self.bonded_atom_pairs:
            d["bondedAtomPairs"] = self.bonded_atom_pairs

        # add user CCD
        if self.user_ccd:
            d["userCCD"] = self.user_ccd

        d["dialect"] = self.dialect
        d["version"] = self.version

        return d

    def write_af3_json(self, filename: str, **kwargs: Any) -> None:
        """Write the job to a JSON file as input for AF3."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, **kwargs)

    @classmethod
    def from_json(cls, filename: str) -> Self:
        """Read a job from a JSON file."""
        obj_match: dict[str, Type[Sequence]] = {
            "protein": ProteinChain,
            "dna": DnaChain,
            "rna": RnaChain,
            "ligand": Ligand,
        }

        with open(filename) as f:
            data = json.load(f)
        job = cls(
            data["name"],
            data["modelSeeds"],
            dialect=data["dialect"],
            version=data["version"],
        )

        if bonded_pairs := data.get("bondedAtomPairs"):
            for pair in bonded_pairs:
                job.add_bonded_atom_pair(*pair[0], *pair[1])

        if user_ccd := data.get("userCCD"):
            job.user_ccd = user_ccd

        if sequences := data.get("sequences"):
            for sequence in sequences:
                ((seq_type, seq_data),) = sequence.items()
                job.sequences.append(obj_match[seq_type].from_dict(seq_data))

        return job

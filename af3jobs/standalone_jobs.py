"""
Module for defining molecular components, modifications, and job configurations for AlphaFold 3.

This module provides data structures representing various molecular elements, including protein chains, nucleotide
chains (DNA/RNA), ligands/ions, and associated modifications. It is tailored for constructing job configurations
for AlphaFold 3 in JSON format.

The `Job` class is the main container for combining chains, ligands/ions, and modifications that can be converted
to JSON as input for AlphaFold 3.

It follows the JSON schema defined in the AlphaFold 3 documentation (as of November 2024):
https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from random import randint
from typing import Any, Self
import warnings

from .utils import chain_id


@dataclass
class SequenceModification:
    """Represents a sequence modifications."""

    mod_type: str  # modification type
    position: int  # position of the modification (1-based)


@dataclass
class Template:
    """Represents a template for a protein chain."""

    mmcif: str
    query_indices: list[int]
    template_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "mmcif": self.mmcif,
            "queryIndices": self.query_indices,
            "templateIndices": self.template_indices,
        }
        return d


@dataclass
class Chain:
    """Base class for protein and nucleotide chains."""

    ids: str | list[str]
    sequence: str
    modifications: list[SequenceModification] = field(default_factory=list)

    def add_modification(self, mod_type: str, position: int) -> Self:
        """Add a modification to the chain."""
        mod = SequenceModification(mod_type, position)
        self.modifications.append(mod)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.ids if len(self.ids) > 1 else self.ids[0],
            "sequence": self.sequence,
        }


@dataclass
class ProteinChain(Chain):
    """Represents a protein chain in the job definition."""

    unpaired_msa: str | None = None
    paired_msa: str | None = None
    templates: list = field(default_factory=list)

    def add_template(
        self, mmcif: str, query_indices: list[int], template_indices: list[int]
    ) -> Self:
        """Add a template to the protein chain."""
        if len(query_indices) != len(template_indices):
            raise ValueError("Query and template indices must have the same length.")
        if not mmcif:
            raise ValueError("Empty mmCIF template string.")
        template = Template(mmcif, query_indices, template_indices)
        self.templates.append(template)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"ptmType": mod.mod_type, "ptmPosition": mod.position}
                for mod in self.modifications
            ]
        if self.unpaired_msa is not None:
            d["unpairedMsa"] = self.unpaired_msa
        if self.paired_msa is not None:
            d["pairedMsa"] = self.paired_msa
        if self.templates:
            d["templates"] = [template.to_dict() for template in self.templates]
        return {"protein": d}


class DnaChain(Chain):
    """Represents a DNA chain in the job definition."""

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]
        return {"dna": d}


@dataclass
class RnaChain(Chain):
    """Represents an RNA chain in the job definition."""

    unpaired_msa: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]
        if self.unpaired_msa is not None:
            d["unpairedMsa"] = self.unpaired_msa
        return {"rna": d}


@dataclass
class Ligand:
    """Represents a ligand or an ion in the job definition."""

    ids: str | list[str]
    ccd_codes: None | str | list[str] = None
    smiles: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.ids if len(self.ids) > 1 else self.ids[0]}
        # if a SMILES string is provided, use it; otherwise, use CCD codes
        if self.smiles:
            d["smiles"] = self.smiles
        else:
            d["ccdCodes"] = self.ccd_codes
        return {"ligand": d}


@dataclass
class Job:
    """Represents an AlphaFold3 job with methods to add entities."""

    name: str
    model_seeds: list[int] = field(default_factory=list)
    sequences: list[Chain | Ligand] = field(default_factory=list)
    bonded_atom_pairs: list = field(default_factory=list)
    user_ccd: str = ""
    dialect: str = "alphafold3"
    version: str = 1

    def __post_init__(self):
        """Post-initialization method."""
        if not self.model_seeds:
            # generate a random model seed
            self.model_seeds = [randint(2**5, 2**30)]

        # Initialize the chain ID generator
        self._chain_ids = chain_id()

    def _check_ids(self) -> None:
        """Check for duplicate IDs."""
        seq_ids = []
        for seq in self.sequences:
            seq_ids.extend(seq.ids)
        if len(seq_ids) != len(set(seq_ids)):
            raise ValueError(f"Duplicate chain IDs found: {seq_ids}")

    def _get_ids(self, ids, count) -> list[str]:
        if ids is None:
            if count >= 1:
                ids = list(islice(self._chain_ids, count))
            else:
                raise ValueError(
                    "Number of chains or ligands must be greater than zero."
                )
        else:
            if isinstance(ids, str):
                ids = [ids]
            elif not isinstance(ids, list):
                raise TypeError("IDs must be a string or a list of strings.")
        return ids

    def _add_chain(
        self,
        chain_type: str,
        sequence: str,
        count: int = 1,
        ids: None | str | list[str] = None,
    ) -> ProteinChain | DnaChain | RnaChain:
        """Create a chain object depending on 'chain_type'."""
        ids = self._get_ids(ids, count)
        match chain_type:
            case "protein":
                chn = ProteinChain(ids, sequence)
            case "dna":
                chn = DnaChain(ids, sequence)
            case "rna":
                chn = RnaChain(ids, sequence)
            case _:
                raise ValueError(f"Invalid chain type: {chain_type}")
        self.sequences.append(chn)
        return chn

    def add_protein_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> ProteinChain:
        """Add a protein chain to the job."""
        return self._add_chain("protein", sequence, count, ids)

    def add_dna_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> DnaChain:
        """Add a DNA chain to the job."""
        return self._add_chain("dna", sequence, count, ids)

    def add_rna_chain(
        self, sequence: str, count: int = 1, ids: None | str | list[str] = None
    ) -> RnaChain:
        """Add an RNA chain to the job."""
        return self._add_chain("rna", sequence, count, ids)

    def add_ligand(
        self,
        ccd_codes: str | list[str] | None = None,
        smiles: str | None = None,
        count: int = 1,
        ids: None | str | list[str] = None,
    ) -> Ligand:
        """Add a ligand to the job."""
        if ccd_codes is None and smiles is None:
            raise ValueError("Either CCD codes or SMILES string must be provided.")
        if ccd_codes is not None and smiles is not None:
            warnings.warn(
                "`ccd_codes` and `smiles` are given - they are mutually exclusive - will be using smiles"
            )
        ids = self._get_ids(ids, count)
        if smiles:
            ligand = Ligand(ids, smiles=smiles)
        else:
            if isinstance(ccd_codes, str):
                ccd_codes = [ccd_codes]
            ligand = Ligand(ids, ccd_codes=ccd_codes)
        self.sequences.append(ligand)
        return ligand

    def to_dict(self) -> dict[str, Any]:
        """Convert the Job to a dictionary suitable for JSON serialization."""
        d = {
            "name": self.name,
            "modelSeeds": self.model_seeds,
        }

        # add sequences / ligands / ions
        if self.sequences:
            self._check_ids()
            d["sequences"] = [seq.to_dict() for seq in self.sequences]
        else:
            raise ValueError("Empty list of sequences.")

        # add bonded atom pairs
        if self.bonded_atom_pairs:
            d["bondedAtomPairs"] = self.bonded_atom_pairs

        # add user CCD
        if self.user_ccd:
            d["userCcd"] = self.user_ccd

        d["dialect"] = self.dialect
        d["version"] = self.version

        return d


if __name__ == "__main__":
    import json

    # Create a new job
    job = Job(name="Sample AlphaFold Job", model_seeds=[42])
    # Add a protein chain with glycans and modifications
    protein_chain = job.add_protein_chain(sequence="MVLSEGEWQLVLHVWAKVEA", count=2)
    protein_chain.add_modification(mod_type="HY3", position=1)

    # Add a DNA chain with modifications
    dna_chain = job.add_dna_chain(sequence="GATTACA", count=1)
    dna_chain.add_modification(mod_type="6OG", position=1)

    # Add an RNA chain with modifications
    rna_chain = job.add_rna_chain(sequence="GUAC", count=1)
    rna_chain.add_modification(mod_type="2MG", position=3)

    # Add a ligand and an ion
    job.add_ligand(ccd_codes="HEM", count=2)

    # Convert the job to a dictionary and print it
    job_dict = job.to_dict()
    print("Job as dictionary:\n", job_dict)

    # Save the job as a JSON file
    with open("job_request.json", "w") as json_file:
        json.dump(job_dict, json_file, indent=2)

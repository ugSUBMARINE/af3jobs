"""
This module provides data structures representing various molecular elements, including protein chains, nucleotide
chains (DNA/RNA), ligands/ions, and associated modifications. It is tailored for constructing job configurations
for AlphaFold 3 in JSON format.
"""

from dataclasses import dataclass, field
from typing import Any, Self

from .utils import get_msa_from_a3m, get_msa_from_json, get_templates_from_json


# Base class for sequence modifications
@dataclass
class SequenceModification:
    """Base class for sequence modifications."""

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
    templates: list[Template] = field(default_factory=list)

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

    def add_templates_from_json(self, json_file: str) -> Self:
        """Obtain templates from a JSON file generated by AF3 (<job_name>_data.json)."""
        templates = get_templates_from_json(json_file, self.sequence)
        if templates is None:
            raise ValueError(
                f"No templates found for protein chain(s) {self.ids} in {json_file}."
            )
        for template in templates:
            self.add_template(*template.values())
        return self

    def add_msa_from_json(self, json_file: str) -> Self:
        """Obtain MSA from a JSON file generated by AF3 (<job_name>_data.json)."""
        msa = get_msa_from_json(json_file, self.sequence)
        if msa is None:
            raise ValueError(
                f"No MSA found for protein chain(s) {self.ids} in {json_file}."
            )
        self.unpaired_msa = msa
        self.paired_msa = ""
        return self

    def add_msa_from_a3m(self, a3m_file: str) -> Self:
        """Obtain MSA from an A3M file."""
        msa = get_msa_from_a3m(a3m_file, self.sequence)
        if msa is None:
            raise ValueError(
                f"Protein sequence of chain(s) {self.ids} not the query sequence in {a3m_file}."
            )
        self.unpaired_msa = msa
        self.paired_msa = ""
        return self

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"ptmType": mod.mod_type, "ptmPosition": mod.position}
                for mod in self.modifications
            ]

        # check the combination of MSA and template data
        if self.unpaired_msa is None and self.paired_msa is not None:
            # if only paired MSA is provided, raise an error
            # 0, 1, 0  ERROR
            # 0, 1, 1  ERROR
            raise ValueError(
                f"Only 'paired_msa' is provided for protein chain(s) {self.ids}."
            )
        if self.templates:
            # if templates are provided, unpaired MSA and paired MSA must be set, but can be empty
            # 0, 0, 1  OK
            # 1, 1, 1  OK
            # 1, 0, 1  OK
            d["unpairedMsa"] = self.unpaired_msa or ""
            d["pairedMsa"] = self.paired_msa or ""
            d["templates"] = [template.to_dict() for template in self.templates]
        elif self.unpaired_msa is not None:
            # if only unpaired MSA is provided, paired MSA must be set, but can be empty
            # 1, 0, 0  OK
            # 1, 1, 0  OK
            d["unpairedMsa"] = self.unpaired_msa
            d["pairedMsa"] = self.paired_msa or ""
            d["templates"] = []
        else:
            # neither MSA nor templates are provided, do nothing
            # 0, 0, 0  OK
            pass

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

    def add_msa_from_json(self, json_file: str) -> Self:
        """Obtain MSA from a JSON file generated by AF3 (<job_name>_data.json)."""
        msa = get_msa_from_json(json_file, self.sequence)
        if msa is None:
            raise ValueError(
                f"No MSA found for RNA chain(s) {self.ids} in {json_file}."
            )
        self.unpaired_msa = msa
        return self

    def add_msa_from_a3m(self, a3m_file: str) -> Self:
        """Obtain MSA from an A3M file."""
        msa = get_msa_from_a3m(a3m_file, self.sequence)
        if msa is None:
            raise ValueError(
                f"RNA sequence of chain(s) {self.ids} not the query sequence in {a3m_file}."
            )
        self.unpaired_msa = msa
        return self

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

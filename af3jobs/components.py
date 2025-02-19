"""
This module provides data structures representing various molecular elements, including protein chains, nucleotide
chains (DNA/RNA), ligands/ions, and associated modifications. It is tailored for constructing job configurations
for AlphaFold 3 in JSON format.
"""

from __future__ import annotations

import warnings
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

    mmcif: str | None  # mmCIF template string
    query_indices: list[int]
    template_indices: list[int]
    mmcif_path: str | None = None  # path to the mmCIF file

    def __post_init__(self):
        if self.mmcif and self.mmcif_path:
            raise ValueError(
                "Both 'mmcif' and 'mmcif_path' are provided for the template."
            )
        if not self.mmcif and not self.mmcif_path:
            raise ValueError(
                "Neither 'mmcif' nor 'mmcif_path' is provided for the template."
            )
        if len(self.query_indices) != len(self.template_indices):
            raise ValueError("Query and template indices must have the same length.")

    def to_dict(self, version: int) -> dict[str, Any]:
        if self.mmcif:
            mmcif_key = "mmcif"
            mmcif_value = self.mmcif
        elif self.mmcif_path and version > 1:
            mmcif_key = "mmcifPath"
            mmcif_value = self.mmcif_path
        else:
            raise ValueError("mmCIF string is required for AF3 input file version 1.")

        return {
            mmcif_key: mmcif_value,
            "queryIndices": self.query_indices,
            "templateIndices": self.template_indices,
        }


@dataclass
class Chain:
    """Base class for protein and nucleotide chains."""

    ids: str | list[str]
    sequence: str
    modifications: list[SequenceModification] = field(default_factory=list)
    unpaired_msa: str | None = None
    unpaired_msa_path: str | None = None
    paired_msa: str | None = None
    paired_msa_path: str | None = None

    def add_modification(self, mod_type: str, position: int) -> Self:
        """Add a modification to the chain."""
        mod = SequenceModification(mod_type, position)
        self.modifications.append(mod)
        return self

    def to_dict(self, *args: Any) -> dict[str, Any]:
        return {
            "id": self.ids if len(self.ids) > 1 else self.ids[0],
            "sequence": self.sequence,
        }

    def add_msa_from_json(self, json_file: str) -> Self:
        """Add unpaired MSA from a JSON file generated by AF3 (<job_name>_data.json)."""
        if not isinstance(self, ProteinChain) and not isinstance(self, RnaChain):
            warnings.warn("MSA is ignored for non-protein and non-RNA chains!")
        else:
            msa = get_msa_from_json(json_file, self.sequence)
            if msa is None:
                raise ValueError(
                    f"No MSA found for protein/RNA chain(s) {self.ids} in {json_file}."
                )
            self.unpaired_msa = msa
            if self.paired_msa is None:
                self.paired_msa = ""
        return self

    def add_paired_msa_from_json(self, json_file: str) -> Self:
        """Add paired MSA from a JSON file generated by AF3 (<job_name>_data.json)."""
        if not isinstance(self, ProteinChain):
            warnings.warn("Paired MSA is ignored for non-protein chains!")
        else:
            msa = get_msa_from_json(json_file, self.sequence, paired=True)
            if msa is None:
                raise ValueError(
                    f"No paired MSA found for protein chain(s) {self.ids} in {json_file}."
                )
            self.paired_msa = msa
            if self.unpaired_msa is None:
                self.unpaired_msa = ""
        return self

    def add_msa_from_a3m(self, a3m_file: str) -> Self:
        """Add unpaired MSA from an A3M file."""
        if not isinstance(self, ProteinChain) and not isinstance(self, RnaChain):
            warnings.warn("MSA is ignored for non-protein and non-RNA chains!")
        else:
            msa = get_msa_from_a3m(a3m_file, self.sequence)
            if msa is None:
                raise ValueError(
                    f"Protein/RNA sequence of chain(s) {self.ids} is not the query sequence in {a3m_file}."
                )
            self.unpaired_msa = msa
            if self.paired_msa is None:
                self.paired_msa = ""
        return self

    def set_unpaired_msa_path(self, path: str) -> Self:
        """Set the path to the unpaired MSA file. ONLY for AF3 input file version >= 2."""
        self.unpaired_msa_path = path
        return self

    def set_paired_msa_path(self, path: str) -> Self:
        """Set the path to the paired MSA file. ONLY for AF3 input file version >= 2."""
        self.paired_msa_path = path
        return self

    def set_unpaired_msa(self, msa: str) -> Self:
        """Set the unpaired MSA."""
        self.unpaired_msa = msa
        return self

    def set_paired_msa(self, msa: str) -> Self:
        """Set the paired MSA."""
        self.paired_msa = msa
        return self


@dataclass
class ProteinChain(Chain):
    """Represents a protein chain in the job definition."""

    templates: None | list[Template] = None

    def add_template(
        self, mmcif: str, query_indices: list[int], template_indices: list[int]
    ) -> Self:
        """Add a template to the protein chain."""
        template = Template(mmcif, query_indices, template_indices)
        self.templates.append(template)
        return self

    def add_template_mmcif_path(
        self, mmcif_path: str, query_indices: list[int], template_indices: list[int]
    ) -> Self:
        """Add a template to the protein chain using a path to an mmCIF file. ONLY for AF3 input file version >= 2."""
        template = Template(None, query_indices, template_indices, mmcif_path)
        if self.templates is None:
            self.templates = []
        self.templates.append(template)
        return self

    def add_templates_from_json(self, json_file: str) -> Self:
        """Add templates from a JSON file generated by AF3 (<job_name>_data.json)."""
        templates = get_templates_from_json(json_file, self.sequence)
        if templates is None:
            raise ValueError(
                f"No templates found for protein chain(s) {self.ids} in {json_file}."
            )
        if self.templates is None:
            self.templates = []
        for template in templates:
            self.add_template(*template.values())
        return self

    def set_empty_templates(self) -> Self:
        """Set an empty list of templates to ensure that no template search is performed."""
        self.templates = []
        return self

    def to_dict(self, version: int = 2) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"ptmType": mod.mod_type, "ptmPosition": mod.position}
                for mod in self.modifications
            ]

        # handling MSA and MSA path
        if self.unpaired_msa is not None and self.unpaired_msa_path is not None:
            raise ValueError(
                f"Both 'unpaired_msa' and 'unpaired_msa_path' are provided for protein chain(s) {self.ids}."
            )
        if self.paired_msa is not None and self.paired_msa_path is not None:
            raise ValueError(
                f"Both 'paired_msa' and 'paired_msa_path' are provided for protein chain(s) {self.ids}."
            )

        if self.unpaired_msa is not None:
            # case 1 0
            # case 1 1
            d["unpairedMsa"] = self.unpaired_msa
            d["pairedMsa"] = self.paired_msa or ""
        elif self.paired_msa is not None:
            # case 0 1
            d["unpairedMsa"] = ""
            d["pairedMsa"] = self.paired_msa
        elif self.unpaired_msa_path is not None and version > 1:
            d["unpairedMsaPath"] = self.unpaired_msa_path
            d["pairedMsaPath"] = self.paired_msa_path or ""
        elif self.paired_msa_path is not None and version > 1:
            d["unpairedMsaPath"] = ""
            d["pairedMsaPath"] = self.paired_msa_path

        # if a (empty) list of templates are provided, no temple search is performed
        if self.templates is not None:
            d["templates"] = [template.to_dict(version) for template in self.templates]

        return {"protein": d}


class DnaChain(Chain):
    """Represents a DNA chain in the job definition."""

    def to_dict(self, *args: Any) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]
        return {"dna": d}


class RnaChain(Chain):
    """Represents an RNA chain in the job definition."""

    def to_dict(self, version: int = 2) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]

        # handling MSA and MSA path
        if self.unpaired_msa is not None and self.unpaired_msa_path is not None:
            raise ValueError(
                f"Both 'unpaired_msa' and 'unpaired_msa_path' are provided for RNA chain(s) {self.ids}."
            )

        if self.unpaired_msa is not None:
            d["unpairedMsa"] = self.unpaired_msa
        elif self.unpaired_msa_path is not None and version > 1:
            d["unpairedMsaPath"] = self.unpaired_msa_path

        return {"rna": d}


@dataclass
class Ligand:
    """Represents a ligand or an ion in the job definition."""

    ids: str | list[str]
    ccd_codes: None | str | list[str] = None
    smiles: str = ""

    def to_dict(self, *args: Any) -> dict[str, Any]:
        d = {"id": self.ids if len(self.ids) > 1 else self.ids[0]}
        # if a SMILES string is provided, use it; otherwise, use CCD codes
        if self.smiles:
            d["smiles"] = self.smiles
        else:
            d["ccdCodes"] = self.ccd_codes
        return {"ligand": d}

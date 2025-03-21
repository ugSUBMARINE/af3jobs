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


@dataclass
class SequenceModification:
    """Base class for sequence modifications."""

    mod_type: str  # modification type
    position: int  # position of the modification (1-based)

    def __str__(self) -> str:
        return f"   Modification: {self.mod_type} at position {self.position}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            data.get("ptmType") or data.get("modificationType"),  # type: ignore
            data.get("ptmPosition") or data.get("basePosition"),  # type: ignore
        )


@dataclass
class Template:
    """Represents a template for a protein chain."""

    mmcif: str | None  # mmCIF template string
    query_indices: list[int]
    template_indices: list[int]
    mmcif_path: str | None = None  # path to the mmCIF file

    def __post_init__(self) -> None:
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

    def __str__(self) -> str:
        if self.mmcif:
            data = [
                line.split("data_")[-1]
                for line in self.mmcif.splitlines()
                if line.startswith("data_")
            ]
            prefix = data[0]
        else:
            prefix = f"mmcif_path = {self.mmcif_path!r}"
        return f"{prefix}, {len(self.query_indices)} query indices"

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        mmcif = data.get("mmcif")
        mmcif_path = data.get("mmcifPath")
        query_indices = data.get("queryIndices")
        template_indices = data.get("templateIndices")
        return cls(mmcif, query_indices, template_indices, mmcif_path)  # type: ignore


@dataclass
class Chain:
    """Base class for protein and nucleotide chains."""

    ids: list[str]
    sequence: str
    modifications: list[SequenceModification] = field(default_factory=list)
    unpaired_msa: str | None = None
    unpaired_msa_path: str | None = None
    paired_msa: str | None = None
    paired_msa_path: str | None = None

    def __post_init__(self) -> None:
        """Check for empty sequence."""
        if not self.sequence:
            raise ValueError("Sequence cannot be empty.")

    def __str__(self) -> str:
        if len(self.sequence) > 25:
            seq = f"{self.sequence[:10]}.....{self.sequence[-10:]}"
        else:
            seq = self.sequence
        lines = [
            f"   {'ID' if len(self.ids) == 1 else 'IDs'}: {', '.join(self.ids)}",
            f"   Sequence: {seq}, {len(self.sequence)} residues",
        ]
        lines.extend(str(m) for m in self.modifications)
        return "\n".join(lines)

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

    def set_unpaired_msa_path(self, path: str | None) -> Self:
        """Set the path to the unpaired MSA file. ONLY for AF3 input file version >= 2."""
        self.unpaired_msa_path = path
        return self

    def set_paired_msa_path(self, path: str | None) -> Self:
        """Set the path to the paired MSA file. ONLY for AF3 input file version >= 2."""
        self.paired_msa_path = path
        return self

    def set_unpaired_msa(self, msa: str | None) -> Self:
        """Set the unpaired MSA."""
        self.unpaired_msa = msa
        return self

    def set_paired_msa(self, msa: str | None) -> Self:
        """Set the paired MSA."""
        self.paired_msa = msa
        return self


@dataclass
class ProteinChain(Chain):
    """Represents a protein chain in the job definition."""

    templates: None | list[Template] = None

    def __post_init__(self) -> None:
        """Check if the protein sequence contains only legal one-letter codes."""
        super().__post_init__()
        one_letter_codes = set("ACDEFGHIKLMNPQRSTVWY")
        diff = set(self.sequence.upper()).difference(one_letter_codes)
        if diff:
            raise ValueError(
                f"Protein sequence contains invalid one-letter codes: {', '.join(repr(b) for b in sorted(diff))}."
            )

    def __str__(self) -> str:
        lines = [
            "++ Protein chain:",
            super().__str__(),
        ]

        if self.unpaired_msa is not None:
            lines.extend(_generate_msa_str(self.unpaired_msa, "Unpaired MSA"))
        elif self.unpaired_msa_path is not None:
            lines.append(f"   Unpaired MSA: path =  {self.unpaired_msa_path!r}")

        if self.paired_msa is not None:
            lines.extend(_generate_msa_str(self.paired_msa, "Paired MSA"))
        elif self.paired_msa_path is not None:
            lines.append(f"   Paired MSA: path = {self.paired_msa_path!r}")

        if self.templates is not None:
            if len(self.templates) > 0:
                lines.append(f"   Template(s): {len(self.templates)}")
                for i, template in enumerate(self.templates, start=1):
                    lines.append(f"{i:7d}: {str(template)}")
            else:
                lines.append(
                    "   Template(s): empty list -> will perform no template search"
                )

        return "\n".join(lines)

    def add_template(
        self, mmcif: str, query_indices: list[int], template_indices: list[int]
    ) -> Self:
        """Add a template to the protein chain."""
        template = Template(mmcif, query_indices, template_indices)
        if self.templates is None:
            self.templates = []
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        ids = data.get("id")
        if isinstance(ids, str):
            ids = [ids]
        protein_chain = (
            cls(ids=ids, sequence=data.get("sequence"))  # type: ignore
            .set_unpaired_msa(data.get("unpairedMsa"))
            .set_unpaired_msa_path(data.get("unpairedMsaPath"))
            .set_paired_msa(data.get("pairedMsa"))
            .set_paired_msa_path(data.get("pairedMsaPath"))
        )

        for mod in data.get("modifications", []):
            protein_chain.modifications.append(SequenceModification.from_dict(mod))

        for template in data.get("templates", []):
            if protein_chain.templates is None:
                protein_chain.templates = []
            protein_chain.templates.append(Template.from_dict(template))

        return protein_chain


class DnaChain(Chain):
    """Represents a DNA chain in the job definition."""

    def __post_init__(self) -> None:
        """Check if the DNA sequence contains only 'A', 'C', 'G', or 'T'."""
        super().__post_init__()
        diff = set(self.sequence.upper()).difference({"A", "C", "G", "T"})
        if diff:
            raise ValueError(
                f"DNA sequence can only contain 'A', 'C', 'G', or 'T'. Found {', '.join(repr(b) for b in sorted(diff))}."
            )

    def __str__(self) -> str:
        lines = [
            "++ DNA chain:",
            super().__str__(),
        ]
        return "\n".join(lines)

    def to_dict(self, *args: Any) -> dict[str, Any]:
        d = super().to_dict()
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]
        return {"dna": d}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        ids = data.get("id")
        if isinstance(ids, str):
            ids = [ids]
        dna_chain = cls(ids=ids, sequence=data.get("sequence"))  # type: ignore

        for mod in data.get("modifications", []):
            dna_chain.modifications.append(SequenceModification.from_dict(mod))

        return dna_chain


class RnaChain(Chain):
    """Represents an RNA chain in the job definition."""

    def __post_init__(self) -> None:
        """Check if the RNA sequence contains only 'A', 'C', 'G', or 'U'."""
        super().__post_init__()
        diff = set(self.sequence.upper()).difference({"A", "C", "G", "U"})
        if diff:
            raise ValueError(
                f"RNA sequence can only contain 'A', 'C', 'G', or 'U'. Found {', '.join(repr(b) for b in sorted(diff))}."
            )

    def __str__(self) -> str:
        lines = [
            "++ RNA chain:",
            super().__str__(),
        ]

        if self.unpaired_msa is not None:
            lines.extend(_generate_msa_str(self.unpaired_msa, "Unpaired MSA"))
        elif self.unpaired_msa_path is not None:
            lines.append(f"   Unpaired MSA: path = {self.unpaired_msa_path!r}")

        return "\n".join(lines)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        ids = data.get("id")
        if isinstance(ids, str):
            ids = [ids]
        rna_chain = (
            cls(ids=ids, sequence=data.get("sequence"))  # type: ignore
            .set_unpaired_msa(data.get("unpairedMsa"))
            .set_unpaired_msa_path(data.get("unpairedMsaPath"))
        )

        for mod in data.get("modifications", []):
            rna_chain.modifications.append(SequenceModification.from_dict(mod))

        return rna_chain


@dataclass
class Ligand:
    """Represents a ligand or an ion in the job definition."""

    ids: list[str]
    ccd_codes: None | list[str] = None
    smiles: str = ""

    def __str__(self) -> str:
        lines = [
            "++ Ligand:",
            f"   {'ID' if len(self.ids) == 1 else 'IDs'}: {', '.join(self.ids)}",
        ]
        if self.smiles:
            lines.append(f"   SMILES: {self.smiles}")
        elif self.ccd_codes:
            lines.append(f"   CCD codes: {', '.join(self.ccd_codes)}")
        return "\n".join(lines)

    def to_dict(self, *args: Any) -> dict[str, Any]:
        d = {"id": self.ids if len(self.ids) > 1 else self.ids[0]}
        # if a SMILES string is provided, use it; otherwise, use CCD codes
        if self.smiles:
            d["smiles"] = self.smiles
        elif self.ccd_codes:
            d["ccdCodes"] = self.ccd_codes
        else:
            # This should not happen, if the ligand object is created via the Job class
            raise RuntimeError(
                "Either SMILES or CCD codes must be provided for the ligand."
            )
        return {"ligand": d}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        ids = data.get("id")
        if isinstance(ids, str):
            ids = [ids]
        return cls(ids, data.get("ccdCodes"), data.get("smiles") or "")  # type: ignore


def _generate_msa_str(msa: str, name: str) -> list[str]:
    """Generate a list of strings for the MSA output"""
    msa_lines = msa.splitlines()
    num_lines = len(msa_lines)
    lines = []
    if num_lines == 0:
        lines.append(f"   {name}: empty string")
    elif num_lines <= 8:
        lines.append(f"   {name}: {num_lines} lines.")
        lines.extend(f"      {line[:60]}" for line in msa_lines)
    else:
        lines.append(f"   {name}: {len(msa_lines)} lines.")
        lines.extend(f"      {line[:60]}" for line in msa_lines[:4])
        lines.append("      ...")
        lines.extend(f"      {line[:60]}" for line in msa_lines[-4:])
    return lines

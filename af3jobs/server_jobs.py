"""
Module for defining molecular components, modifications, and job configurations for AlphaFold 3 JSON input generation.

This module provides data structures representing various molecular elements, including protein chains, nucleotide
chains (DNA/RNA), ligands, ions, and associated modifications. It is tailored for constructing JSON-compatible
job configurations for AlphaFold 3.

The `Job` class is the main container for combining chains, ligands, ions, and modifications, creating a
comprehensive input for AlphaFold 3.

It follows the JSON schema defined in the AlphaFold 3 server documentation (as of November 2024):
https://github.com/google-deepmind/alphafold/blob/main/server/README.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from .components import SequenceModification

# Allowed modifications, ligands and ions
_PROTEIN_MODS = [
    "CCD_SEP",
    "CCD_TPO",
    "CCD_PTR",
    "CCD_NEP",
    "CCD_HIP",
    "CCD_ALY",
    "CCD_MLY",
    "CCD_M3L",
    "CCD_MLZ",
    "CCD_2MR",
    "CCD_AGM",
    "CCD_MCS",
    "CCD_HYP",
    "CCD_HY3",
    "CCD_LYZ",
    "CCD_AHB",
    "CCD_P1L",
    "CCD_SNN",
    "CCD_SNC",
    "CCD_TRF",
    "CCD_KCR",
    "CCD_CIR",
    "CCD_YHA",
]

_DNA_MODS = [
    "CCD_5CM",
    "CCD_C34",
    "CCD_5HC",
    "CCD_6OG",
    "CCD_6MA",
    "CCD_1CC",
    "CCD_8OG",
    "CCD_5FC",
    "CCD_3DR",
]

_RNA_MODS = [
    "CCD_PSU",
    "CCD_5MC",
    "CCD_OMC",
    "CCD_4OC",
    "CCD_5MU",
    "CCD_OMU",
    "CCD_UR3",
    "CCD_A2M",
    "CCD_MA6",
    "CCD_6MZ",
    "CCD_2MG",
    "CCD_OMG",
    "CCD_7MG",
    "CCD_RSQ",
]

_LIGANDS = [
    "CCD_ADP",
    "CCD_ATP",
    "CCD_AMP",
    "CCD_GTP",
    "CCD_GDP",
    "CCD_FAD",
    "CCD_NAD",
    "CCD_NAP",
    "CCD_NDP",
    "CCD_HEM",
    "CCD_HEC",
    "CCD_PLM",
    "CCD_OLA",
    "CCD_MYR",
    "CCD_CIT",
    "CCD_CLA",
    "CCD_CHL",
    "CCD_BCL",
    "CCD_BCB",
]

_IONS = [
    "MG",
    "ZN",
    "CL",
    "CA",
    "NA",
    "MN",
    "K",
    "FE",
    "CU",
    "CO",
]


# ProteinChain class
@dataclass
class ProteinChain:
    """Represents a protein chain in the job definition."""

    sequence: str
    count: int = 1
    glycans: list[SequenceModification] = field(default_factory=list)
    modifications: list[SequenceModification] = field(default_factory=list)
    use_structure_template: bool = True
    max_template_date: str | None = None

    def add_glycan(self, residues: str, position: int) -> Self:
        """Add a glycan to the protein chain."""
        self.glycans.append(SequenceModification(residues, position))
        return self

    def add_modification(self, mod_type: str, position: int) -> Self:
        """Add a modification to the protein chain."""
        if mod_type in _PROTEIN_MODS:
            self.modifications.append(SequenceModification(mod_type, position))
            return self
        else:
            raise ValueError(f"Unknown protein modification type: {mod_type}.")

    def to_dict(self) -> dict[str, Any]:
        d = {"sequence": self.sequence, "count": self.count}
        if self.glycans:
            d["glycans"] = [
                {"residues": glycan.mod_type, "position": glycan.position}
                for glycan in self.glycans
            ]
        if self.modifications:
            d["modifications"] = [
                {"ptmType": mod.mod_type, "ptmPosition": mod.position}
                for mod in self.modifications
            ]
        # new parameters in version 1
        # only include useStructureTemplate if it is False
        if not self.use_structure_template:
            d["useStructureTemplate"] = False
        # only include maxTemplateDate if it has been set explicitly
        if self.max_template_date is not None:
            d["maxTemplateDate"] = self.max_template_date

        return {"proteinChain": d}


# Base NucleotideChain class
@dataclass
class NucleotideChain:
    """Base class for nucleotide chains, allowing modifications."""

    sequence: str
    count: int = 1
    modifications: list[SequenceModification] = field(default_factory=list)

    def add_modification(self, mod_type: str, position: int) -> Self:
        """Add a modification to the nucleotide chain."""
        if mod_type in _DNA_MODS or mod_type in _RNA_MODS:
            self.modifications.append(SequenceModification(mod_type, position))
            return self
        else:
            raise ValueError(f"Unknown nucleotide modification type: {mod_type}.")

    def to_dict(self) -> dict[str, Any]:
        d = {"sequence": self.sequence, "count": self.count}
        if self.modifications:
            d["modifications"] = [
                {"modificationType": mod.mod_type, "basePosition": mod.position}
                for mod in self.modifications
            ]
        return d


# DnaChain and RnaChain classes
class DnaChain(NucleotideChain):
    """Represents a DNA chain in the job definition."""

    def to_dict(self) -> dict[str, Any]:
        return {"dnaSequence": super().to_dict()}


class RnaChain(NucleotideChain):
    """Represents an RNA chain in the job definition."""

    def to_dict(self) -> dict[str, Any]:
        return {"rnaSequence": super().to_dict()}


# Ligand class
@dataclass
class Ligand:
    """Represents a ligand in the job definition."""

    ligand_type: str  # CCD code of the ligand
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"ligand": {"ligand": self.ligand_type, "count": self.count}}


# Ion class
@dataclass
class Ion:
    """Represents an ion in the job definition."""

    ion_type: str  # CCD code of the ligand
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"ion": {"ion": self.ion_type, "count": self.count}}


Sequence = ProteinChain | DnaChain | RnaChain | Ligand | Ion


# Job class to combine all entities and manage them
@dataclass
class Job:
    """Represents an AlphaFold3 job with methods to add entities."""

    name: str
    modelSeeds: list[int] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)
    dialect: str = "alphafoldserver"
    version: int = 1

    def add_protein_chain(
        self,
        sequence: str,
        count: int = 1,
        max_template_date: str | None = None,
        use_structure_template: bool = True,
    ) -> ProteinChain:
        """Add a protein chain to the job."""
        protein_chain = ProteinChain(
            sequence,
            count,
            max_template_date=max_template_date,
            use_structure_template=use_structure_template,
        )
        self.sequences.append(protein_chain)
        return protein_chain

    def add_dna_chain(self, sequence: str, count: int = 1) -> DnaChain:
        """Add a DNA chain to the job."""
        dna_chain = DnaChain(sequence, count)
        self.sequences.append(dna_chain)
        return dna_chain

    def add_rna_chain(self, sequence: str, count: int = 1) -> RnaChain:
        """Add an RNA chain to the job."""
        rna_chain = RnaChain(sequence, count)
        self.sequences.append(rna_chain)
        return rna_chain

    def add_ligand(self, ligand_type: str, count: int = 1) -> Ligand:
        """Add a ligand to the job."""
        if ligand_type in _LIGANDS:
            ligand = Ligand(ligand_type, count)
            self.sequences.append(ligand)
            return ligand
        else:
            raise ValueError(f"Unknown ligand type: {ligand_type}.")

    def add_ion(self, ion_type: str, count: int = 1) -> Ion:
        """Add a ligand to the job."""
        if ion_type in _IONS:
            ion = Ion(ion_type, count)
            self.sequences.append(ion)
            return ion
        else:
            raise ValueError(f"Unknown ion type: {ion_type}.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the Job to a dictionary suitable for JSON serialization."""
        d: dict[str, Any] = {
            "name": self.name,
            "modelSeeds": self.modelSeeds,
        }
        if self.sequences:
            d["sequences"] = [sequence.to_dict() for sequence in self.sequences]
        else:
            raise ValueError("Empty list of sequences.")
        d["dialect"] = self.dialect
        d["version"] = self.version
        return d

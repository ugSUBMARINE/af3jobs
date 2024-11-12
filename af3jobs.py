"""
Module for defining molecular components, modifications, and job configurations for AlphaFold 3 JSON input generation.

This module provides data structures representing various molecular elements, including protein chains, nucleotide
chains (DNA/RNA), ligands, ions, and associated modifications. It is tailored for constructing JSON-compatible
job configurations for AlphaFold 3.

Classes:
    - Glycan: Represents a glycan attached to a protein chain.
        - Attributes:
            - residues (str): Glycan type.
            - position (int): Attachment position on the protein chain (1-based).
        - Methods:
            - to_dict(): Converts the glycan to a dictionary format.

    - ProteinModification: Represents a post-translational modification on a protein chain.
        - Attributes:
            - ptmType (str): CCD code for the protein modification.
            - ptmPosition (int): Position of the modified amino acid (1-based).
        - Methods:
            - to_dict(): Converts the modification to a dictionary format.

    - NucleotideModification: Represents a modification on a nucleotide chain.
        - Attributes:
            - modificationType (str): CCD code for the nucleotide modification.
            - basePosition (int): Position of the modified nucleotide (1-based).
        - Methods:
            - to_dict(): Converts the modification to a dictionary format.

    - ProteinChain: Represents a protein chain with a sequence, optional glycans, and modifications.
        - Attributes:
            - sequence (str): Protein sequence.
            - count (int): Number of identical chains.
            - glycans (list): List of Glycan objects attached to the chain.
            - modifications (list): List of ProteinModification objects on the chain.
        - Methods:
            - add_glycan(residues, position): Adds a glycan to the chain.
            - add_modification(ptmType, ptmPosition): Adds a protein modification.
            - to_dict(): Converts the chain to a dictionary format.

    - NucleotideChain: Represents a nucleotide chain with a sequence and optional modifications.
        - Attributes:
            - sequence (str): Nucleotide sequence.
            - count (int): Number of identical chains.
            - modifications (list): List of NucleotideModification objects on the chain.
        - Methods:
            - add_modification(modificationType, basePosition): Adds a nucleotide modification.
            - to_dict(): Converts the chain to a dictionary format.

    - DnaChain(NucleotideChain): Represents a DNA chain in the job configuration.
        - Methods:
            - to_dict(): Converts the DNA chain to a dictionary format.

    - RnaChain(NucleotideChain): Represents an RNA chain in the job configuration.
        - Methods:
            - to_dict(): Converts the RNA chain to a dictionary format.

    - Ligand: Represents a ligand in the job configuration.
        - Attributes:
            - ligand_type (str): CCD code for the ligand.
            - count (int): Number of ligand molecules.
        - Methods:
            - to_dict(): Converts the ligand to a dictionary format.

    - Ion: Represents an ion in the job configuration.
        - Attributes:
            - ion_type (str): Symbol of the ion.
            - count (int): Number of ion molecules.
        - Methods:
            - to_dict(): Converts the ion to a dictionary format.

    - Job: Manages and organizes all molecular components for an AlphaFold 3 job.
        - Attributes:
            - name (str): Name of the AlphaFold job.
            - modelSeeds (list): List of model seeds for the job.
            - sequences (list): List of ProteinChain, DnaChain, RnaChain, Ligand, or Ion objects.
        - Methods:
            - add_protein_chain(sequence, count): Adds a protein chain.
            - add_dna_chain(sequence, count): Adds a DNA chain.
            - add_rna_chain(sequence, count): Adds an RNA chain.
            - add_ligand(ligand_type, count): Adds a ligand.
            - add_ion(ion_type, count): Adds an ion.
            - to_dict(): Converts the job to a dictionary format for JSON serialization.

Constants:
    - _PROTEIN_MODS: Allowed protein modification CCD codes.
    - _DNA_MODS: Allowed DNA modification CCD codes.
    - _RNA_MODS: Allowed RNA modification CCD codes.
    - _LIGANDS: Allowed ligands.
    - _IONS: Allowed ions.

The `Job` class is the main container for combining chains, ligands, ions, and modifications, creating a
comprehensive input for AlphaFold 3.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Self

__author__ = "Karl Gruber"
__email__ = "karl.gruber@uni-graz.at"
__version__ = "0.1.0"
__date__ = "2024-11-11"

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


# Glycan class to represent glycosylation on protein chains
@dataclass
class Glycan:
    """Represents a glycan attached to a protein chain."""

    residues: str  # Glycan type
    position: int  # Position of glycan attachment (1-based)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Modification classes to represent modifications on protein, DNA, or RNA chains
@dataclass
class ProteinModification:
    """Represents a post-translational modification on a protein chain."""

    ptmType: str  # CCD code for modification
    ptmPosition: int  # Position of the modified amino acid (1-based)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NucleotideModification:
    """Represents a modification on nucleotide chain."""

    modificationType: str  # CCD code for modification
    basePosition: int  # Position of the modified nucleotide (1-based)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ProteinChain class
@dataclass
class ProteinChain:
    """Represents a protein chain in the job definition."""

    sequence: str
    count: int = 1
    glycans: list[Glycan] = field(default_factory=list)
    modifications: list[ProteinModification] = field(default_factory=list)

    def add_glycan(self, residues: str, position: int) -> Self:
        """Add a glycan to the protein chain."""
        self.glycans.append(Glycan(residues, position))
        return self

    def add_modification(self, ptmType: str, ptmPosition: int) -> Self:
        """Add a modification to the protein chain."""
        if ptmType in _PROTEIN_MODS:
            self.modifications.append(ProteinModification(ptmType, ptmPosition))
            return self
        else:
            raise ValueError(f"Unknown protein modification type: {ptmType}.")

    def to_dict(self) -> dict[str, Any]:
        d = {"sequence": self.sequence, "count": self.count}
        if self.glycans:
            d["glycans"] = [glycan.to_dict() for glycan in self.glycans]
        if self.modifications:
            d["modifications"] = [mod.to_dict() for mod in self.modifications]
        return {"proteinChain": d}


# Base NucleotideChain class
@dataclass
class NucleotideChain:
    """Base class for nucleotide chains, allowing modifications."""

    sequence: str
    count: int = 1
    modifications: list[NucleotideModification] = field(default_factory=list)

    def add_modification(self, modificationType: str, basePosition: int) -> Self:
        """Add a modification to the nucleotide chain."""
        if modificationType in _DNA_MODS or modificationType in _RNA_MODS:
            self.modifications.append(
                NucleotideModification(modificationType, basePosition)
            )
            return self
        else:
            raise ValueError(
                f"Unknown nucleotide modification type: {modificationType}."
            )

    def to_dict(self) -> dict[str, Any]:
        d = {"sequence": self.sequence, "count": self.count}
        if self.modifications:
            d["modifications"] = [mod.to_dict() for mod in self.modifications]
        return d


# DnaChain and RnaChain classes
@dataclass
class DnaChain(NucleotideChain):
    """Represents a DNA chain in the job definition."""

    def to_dict(self) -> dict[str, Any]:
        return {"dnaSequence": super().to_dict()}


@dataclass
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


# Job class to combine all entities and manage them
@dataclass
class Job:
    """Represents an AlphaFold3 job with methods to add entities."""

    name: str
    modelSeeds: list[int] = field(default_factory=list)
    sequences: list[ProteinChain | DnaChain | RnaChain | Ligand | Ion] = field(
        default_factory=list
    )

    def add_protein_chain(self, sequence: str, count: int = 1) -> ProteinChain:
        """Add a protein chain to the job."""
        protein_chain = ProteinChain(sequence, count)
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
            raise ValueError(f"Unknown ion type: {ligand_type}.")

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
        d = {
            "name": self.name,
            "modelSeeds": self.modelSeeds,
        }
        if self.sequences:
            d["sequences"] = [sequence.to_dict() for sequence in self.sequences]
        else:
            raise ValueError("Empty list of sequences.")
        return d


if __name__ == "__main__":
    import json

    # Create a new job
    job = Job(name="Sample AlphaFold Job")

    # Add a protein chain with glycans and modifications
    protein_chain = job.add_protein_chain(sequence="MVLSEGEWQLVLHVWAKVEA", count=2)
    protein_chain.add_glycan(residues="NAG(NAG)(BMA)", position=8)
    protein_chain.add_modification(ptmType="CCD_HY3", ptmPosition=1)

    # Add a DNA chain with modifications
    dna_chain = job.add_dna_chain(sequence="GATTACA", count=1)
    dna_chain.add_modification(modificationType="CCD_6OG", basePosition=1)

    # Add an RNA chain with modifications
    rna_chain = job.add_rna_chain(sequence="GUAC", count=1)
    rna_chain.add_modification(modificationType="CCD_2MG", basePosition=1)

    # Add a ligand and an ion
    job.add_ligand(ligand_type="CCD_ATP", count=1)
    job.add_ion(ion_type="MG", count=2)

    # Convert the job to a dictionary and print it
    job_dict = job.to_dict()
    print("Job as dictionary:", job_dict)

    # Save the job as a JSON file
    with open("job_request.json", "w") as json_file:
        json.dump([job_dict], json_file, indent=4)

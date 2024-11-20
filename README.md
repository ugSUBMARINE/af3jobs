# af3jobs

`af3jobs` is a Python package designed to streamline the process of creating JSON input files for AlphaFold 3.
It provides data structures and tools for defining molecular components, modifications, and job
configurations compatible with both the [AlphaFold 3 server](https://alphafoldserver.com/welcome) and the [standalone versions](https://github.com/google-deepmind/alphafold3).

## Features

- Define **protein**, **DNA**, and **RNA chains** with sequence modifications (e.g., PTMs, nucleotide modifications).
- Add **ligands**, **ions**, and other small molecules with CCD codes or SMILES strings.
- Generate **JSON input files**.

## Installation

Clone the repository and install the package using pip:

```bash
git clone https://github.com/your_username/af3jobs.git
cd af3jobs
pip install .
```

## Usage

### Server Jobs Example

The `server_jobs.py` module focuses on preparing jobs for the AlphaFold 3 server.

```python
import json
from af3jobs.server_jobs import Job

# Create a new job
job = Job(name="Sample AlphaFold Job")

# Add a protein chain with glycans and modifications
protein_chain = job.add_protein_chain(sequence="MVLSEGEWQLVLHVWAKVEA", count=2)
protein_chain.add_glycan(residues="NAG(NAG)(BMA)", position=8)
protein_chain.add_modification(mod_type="CCD_HY3", position=1)

# Add a DNA chain with modifications
dna_chain = job.add_dna_chain(sequence="GATTACA", count=1)
dna_chain.add_modification(mod_type="CCD_6OG", position=1)

# Add a ligand and an ion
job.add_ligand(ligand_type="CCD_ATP", count=1)
job.add_ion(ion_type="MG", count=2)

# Export to JSON
with open("server_job.json", "w") as f:
    json.dump([job.to_dict()], f, indent=4)
```

### Standalone Jobs Example

The `standalone_jobs.py` module focuses on preparing jobs for the standalone AlphaFold 3 application.

```python
import json
from af3jobs.standalone_jobs import Job

# Create a new job
job = Job(name="Sample AlphaFold Job", model_seeds=[42])

# Add a protein chain with modifications
protein_chain = job.add_protein_chain(sequence="MVLSEGEWQLVLHVWAKVEA", count=2)
protein_chain.add_modification(mod_type="HY3", position=1)

# Add a DNA chain with modifications
dna_chain = job.add_dna_chain(sequence="GATTACA", count=1)
dna_chain.add_modification(mod_type="6OG", position=1)

# Add two heme cofactors
job.add_ligand(ccd_codes="HEM", ids=["X", "Y"])

# Export to JSON
with open("standalone_job.json", "w") as f:
    json.dump(job.to_dict(), f, indent=4)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [AlphaFold Server Documentation](https://github.com/google-deepmind/alphafold/blob/main/server/README.md)
- [AlphaFold3 input file format](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)

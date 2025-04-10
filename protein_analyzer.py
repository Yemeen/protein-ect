from pathlib import Path
from Bio import PDB
import requests
import pandas as pd
import numpy as np
from ect import ECT, EmbeddedGraph


class ProteinAnalyzer:
    """Handles protein structure analysis and visualization"""

    def __init__(self, pdb_dir="pdb_files"):
        self.pdb_dir = pdb_dir
        Path(pdb_dir).mkdir(exist_ok=True)

    def load_protein_data(self):
        """Loads protein data from CSV file"""
        return pd.read_csv("secuencia_2025_hoja_1.csv")

    def download_pdb(self, pdb_code):
        """Downloads PDB file for given code"""
        pdb_code = pdb_code.lower()
        output_path = Path(self.pdb_dir) / f"{pdb_code}.pdb"

        if output_path.exists():
            return output_path

        url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
        response = requests.get(url)

        if response.status_code == 200:
            output_path.write_bytes(response.content)
            return output_path
        return None

    def select_representative_chain(self, model):
        """Selects the most representative chain from a model"""
        best_chain = None
        max_residues = 0

        for chain in model.get_chains():
            residue_count = len(list(chain.get_residues()))
            if residue_count > max_residues:
                max_residues = residue_count
                best_chain = chain.id

        return best_chain

    def get_complete_residues(self, structure, chain_id=None):
        """Returns only complete residues with all backbone atoms"""
        complete_residues = []
        for model in structure:
            for chain in model:
                if chain_id and chain.id != chain_id:
                    continue
                for residue in chain:
                    if {"N", "CA", "C", "O"}.issubset(set(residue.child_dict.keys())):
                        complete_residues.append(residue)
        return complete_residues

    def get_seqres_ranges(self, structure):
        """Gets the expected residue ranges from structure"""
        ranges = {}

        for chain in structure[0]:
            residues = list(chain.get_residues())
            if residues:
                first_res = residues[0].id[1]
                last_res = residues[-1].id[1]

                seqres_len = None
                if "seqres" in structure.header:
                    seqres = structure.header["seqres"]
                    if chain.id in seqres:
                        seqres_len = len(seqres[chain.id])

                if seqres_len is None:
                    seqres_len = len(residues)

                ranges[chain.id] = {
                    "seqres_length": seqres_len,
                    "start": first_res,
                    "end": last_res,
                    "actual_residues": len(residues),
                }

        return ranges

    def analyze_structure(self, pdb_file, single_chain=True):
        """Analyzes protein structure for residues, atoms, and secondary structure"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]

        chain_id = self.select_representative_chain(model) if single_chain else None

        seqres_ranges = self.get_seqres_ranges(structure)

        residue_count = 0
        atom_count = 0
        helix_residues = 0
        sheet_residues = 0
        problematic_residues = []
        missing_atoms = {}
        outside_range_residues = []

        for model in structure:
            for chain in model:
                if single_chain and chain.id != chain_id:
                    continue

                chain_range = seqres_ranges.get(chain.id, {})
                expected_start = chain_range.get("start", None)
                expected_end = chain_range.get("end", None)

                residue_count += len(list(chain))
                for residue in chain:
                    res_num = residue.id[1]

                    if expected_start is not None and expected_end is not None:
                        if res_num < expected_start or res_num > expected_end:
                            outside_range_residues.append(f"{chain.id}{res_num}")
                            continue

                    atom_count += len(residue)

                    missing = {"N", "CA", "C", "O"} - set(residue.child_dict.keys())
                    if missing:
                        res_id = f"{chain.id}{res_num}"
                        problematic_residues.append(res_id)
                        missing_atoms[res_id] = list(missing)

        complete_residues = self.get_complete_residues(structure[0], chain_id)

        new_structure = PDB.Structure.Structure("filtered")
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)

        chain_residues = {}
        for res in complete_residues:
            if res.parent.id not in chain_residues:
                chain_residues[res.parent.id] = []
            chain_residues[res.parent.id].append(res)

        for chain_id, residues in chain_residues.items():
            new_chain = PDB.Chain.Chain(chain_id)
            new_model.add(new_chain)
            for res in residues:
                new_chain.add(res.copy())

        dssp_error = None
        try:
            tmp_pdb = Path(pdb_file).parent / "tmp_complete.pdb"
            io = PDB.PDBIO()
            io.set_structure(new_structure)
            io.save(str(tmp_pdb))

            dssp = PDB.DSSP(new_structure[0], str(tmp_pdb), dssp="mkdssp")

            tmp_pdb.unlink()

            for key, residue in dssp.property_dict.items():
                if residue[2] in ["H", "G", "I"]:
                    helix_residues += 1
                elif residue[2] in ["E", "B"]:
                    sheet_residues += 1
        except Exception as e:
            helix_residues = None
            sheet_residues = None
            dssp_error = str(e)

        return {
            "residues": residue_count,
            "atoms": atom_count,
            "helix": helix_residues,
            "sheet": sheet_residues,
            "selected_chain": chain_id if single_chain else None,
            "problematic_residues": problematic_residues,
            "missing_atoms": missing_atoms,
            "dssp_error": dssp_error,
            "complete_residue_count": len(complete_residues),
            "seqres_ranges": seqres_ranges,
            "outside_range_residues": outside_range_residues,
        }

    def get_structure_info(self, pdb_file, single_chain=True):
        """Gets detailed information about protein structure"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]

        chain_id = self.select_representative_chain(model) if single_chain else None

        info = {
            "chains": [],
            "resolution": None,
            "r_factor": None,
            "space_group": None,
            "experimental_method": None,
            "deposition_date": None,
            "selected_chain": chain_id if single_chain else None,
        }

        header = parser.get_structure("protein", pdb_file).header

        info["resolution"] = header.get("resolution", "Not available")
        info["structure_method"] = header.get("structure_method", "Not available")
        info["deposition_date"] = header.get("deposition_date", "Not available")
        info["raw_header"] = header
        info["is_xray"] = "x-ray" in str(header.get("structure_method", "")).lower()

        for chain in structure.get_chains():
            if single_chain and chain.id != chain_id:
                continue
            chain_info = {
                "chain_id": chain.id,
                "residue_count": len(list(chain.get_residues())),
                "atom_count": len(list(chain.get_atoms())),
            }
            info["chains"].append(chain_info)

        return info

    def get_backbone_coordinates(self, pdb_file, single_chain=True):
        """Extracts C atoms from ONE chain in FIRST MODEL"""
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]

        chain_id = self.select_representative_chain(model) if single_chain else None

        backbone_coords = []
        for chain in model:
            if single_chain and chain.id != chain_id:
                continue

            for residue in chain:
                if "C" in residue:
                    c_atom = residue["C"]
                    backbone_coords.append(c_atom.get_coord())

        return backbone_coords

    def calculate_ect(
        self, backbone_coords, num_directions=64, num_thresholds=64, radius=None
    ):
        """Calculates ECT from backbone coordinates"""
        if not backbone_coords:
            return None

        G = EmbeddedGraph()

        ect = ECT(
            num_dirs=num_directions, num_thresh=num_thresholds, bound_radius=radius
        )

        for i, coord in enumerate(backbone_coords):
            G.add_node(node_id=i, coord=coord)
            if i > 0:
                G.add_edge(node_id1=i - 1, node_id2=i)

        if radius is None:
            radius = G.get_bounding_radius()

        return ect.calculate(G, override_bound_radius=radius)

    def get_all_structure_methods(self, df):
        """Gets all unique structure determination methods from the PDB files"""
        methods = set()
        for pdb_code in df[df["PDB Code"].notna()]["PDB Code"].unique():
            pdb_file = self.download_pdb(pdb_code)
            if pdb_file:
                info = self.get_structure_info(pdb_file)
                if info["structure_method"] != "Not available":
                    methods.add(info["structure_method"])
        return sorted(list(methods))

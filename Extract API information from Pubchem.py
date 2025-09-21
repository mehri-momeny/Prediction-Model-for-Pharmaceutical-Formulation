# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:09:22 2025

@author: Mehrnaz
"""

import pandas as pd
import pubchempy as pcp
from time import sleep
from tqdm import tqdm

# Load your CSV file
df = pd.read_csv("Data/API_List.csv", encoding='ISO-8859-1')
api_names = df['API'].unique()


def get_pubchem_info(compound_name):
    try:
        compound = pcp.get_compounds(compound_name, 'name')
        if compound:
            c = compound[0]
            return {
                'API': compound_name,
                'CID': c.cid,
                'IUPACName': c.iupac_name,
                'InChIKey': c.inchikey,
                'MolecularFormula': c.molecular_formula,
                'CanonicalSMILES': c.canonical_smiles,
                'MolecularWeight (g/mol)': c.molecular_weight,
                'XLogP3-AA': c.xlogp,
                'HydrogenBondDonorCount': c.h_bond_donor_count,
                'HydrogenBondAcceptorCount': c.h_bond_acceptor_count,
                'RotatableBondCount': c.rotatable_bond_count,
                'TopologicalSurfaceArea (Å²)': c.tpsa,
                'HeavyAtomCount': c.heavy_atom_count,
                'Complexity': c.complexity
            }
        else:
            return {'API': compound_name, 'Error': 'Compound not found'}

    except Exception as e:
        return {'API': compound_name, 'Error': str(e)}

# Collect results with delay
results = []
for name in tqdm(api_names, desc="Api List"):
    results.append(get_pubchem_info(name))
    sleep(0.5)  # Respectful delay

# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv("Data/Mapped_API_to_PubChem_Enhanced.csv", index=False)
print("Enhanced mapping complete. File saved as Mapped_API_to_PubChem_Enhanced.csv")


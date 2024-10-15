from mp_api.client import MPRester
from mp_api.client.routes.materials.summary import SummaryRester
from monty.serialization import loadfn, dumpfn
import pandas as pd
import os
from pymatgen.io.vasp import Poscar  # Import Poscar class

MPI_KEY = "zckQao2291DWQRcIyq96cAwvZH9DAdTy"

def get_data():
    req_fields = ["material_id", "formula_pretty", "nsites", "nelements", "energy_above_hull", "band_gap", "is_magnetic"]
    with SummaryRester(api_key=MPI_KEY) as mpr:
        data = mpr.search(total_magnetization=(None, None), fields=req_fields)

    table = {}
    for field in req_fields:
        print(f"Field: {field}")
        table[field] = [d.__getattribute__(field) for d in data]

    dumpfn(table, "mp_data.json", indent=4)

def get_table():
    # Get CSV
    table = loadfn("mp_data.json")
    df = pd.DataFrame(table)
    # Find those with the field "is_magnetic" is False
    df = df[df["is_magnetic"] == False].sort_values(["energy_above_hull", "band_gap"], ascending=[True, False])
    df.to_csv("mp_data.csv", index=False)

def get_first_column(file_path):
    df = pd.read_csv(file_path, header=None)
    return df[0].tolist()[1:]

def get_mp_structure():
    mp_ids = get_first_column("10142024_is_magnetic_false.csv")
    print(f"Number of materials: {len(mp_ids)}, first 5: {mp_ids[:5]}")
    
    req_fields = ["material_id", "formula_pretty", "structure"]
    output_dir = "./mp_structures"
    os.makedirs(output_dir, exist_ok=True)
    
    with SummaryRester(api_key=MPI_KEY) as mpr:
        # Every 500 materials generate a JSON file
        for i in range(0, len(mp_ids), 500):
            data = mpr.search(material_ids=mp_ids[i:i+500], fields=req_fields)
            table = {}
            for field in req_fields:
                print(f"Field: {field}")
                table[field] = [d.__getattribute__(field) for d in data]
            
            dumpfn(table, os.path.join(output_dir, f"mp_structure_{i}-{i+500}.json"), indent=4)

            # Save VASP input files
            save_vasp_inputs(data, output_dir)

def save_vasp_inputs(data, output_dir):
    for material in data:
        material_id = material.material_id
        structure = material.structure

        # Define a directory for each material
        material_dir = os.path.join(output_dir, material_id)
        os.makedirs(material_dir, exist_ok=True)

        # Create the POSCAR file using pymatgen's Poscar class
        poscar = Poscar(structure)
        poscar_file_path = os.path.join(material_dir, "POSCAR")
        poscar.write_file(poscar_file_path)

        print(f"Saved POSCAR for {material_id} in {material_dir}")

if __name__ == "__main__":
    # Uncomment the following lines as needed
    # get_data()
    # get_table()
    get_mp_structure()

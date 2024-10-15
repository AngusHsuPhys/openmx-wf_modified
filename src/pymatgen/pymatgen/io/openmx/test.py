from ase.calculators.openmx import OpenMX
from ase import Atoms

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Structure

import os

os.chdir(os.path.dirname(__file__))

# add environment variables, export OPENMX_DFT_DATA_PATH=/openmx/DFT_DATA13
os.environ["OPENMX_DFT_DATA_PATH"] = "/workspaces/openmx-wf/ASE/DFT_DATA19"
os.environ["ASE_OPENMX_COMMAND"] = "openmx"

# st = Structure.from_file("/workspaces/openmx-wf/Atomate/launch-wf/C.cif")
st = Structure.from_file("GaAs.vasp")

atom_adaptor = AseAtomsAdaptor()

atoms = atom_adaptor.get_atoms(st)

# potentail
# species = [
#     ["B", "B7.0-s2p2d1", "B_PBE19"],
# ]

# a set of configuration as a dictionary
inputs = dict(
    scf_xctype="GGA-PBE",
    scf_kgrid=(4, 4, 4),
    scf_maxiter=40,
    scf_mixing_type="Simple",
    scf_spinpolarization="off",
    scf_energycutoff=200.0,
    scf_eigenvaluesolver="Band",
    md_type="nomd",

    # definition_of_atomic_species=species,
)

calc = OpenMX(label=f"{st.formula}_openmx", **inputs)

calc.write_input(atoms)



from ase import io
from pymatgen.io.ase import AseAtomsAdaptor
import os 

from monty.serialization import loadfn, dumpfn

# change the directory to the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def read_xyz(filename):
    return io.read(filename, index=':')

def tansform_xyz_to_st(st_ase):
    st_ase_dict = st_ase.todict()
    st = AseAtomsAdaptor.get_structure(st_ase)
    return st, st_ase_dict


def get_st_dict(xyz_file):
    sts = read_xyz(xyz_file)
    new_st_dict = {}
    for i, st_ase in enumerate(sts):
        st, st_ase_dict = tansform_xyz_to_st(st_ase)
        # add a new key st_vasp_dict to the st_ase_dict
        st_ase_dict['st_vasp_dict'] = st.as_dict()
        new_st_dict[i] = st_ase_dict

    dumpfn(new_st_dict, 'initial-train.json')
        


if __name__ == '__main__':
    get_st_dict('20230805-training_data/initial-train.xyz')
    

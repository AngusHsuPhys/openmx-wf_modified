from atomate.openmx.fireworks.core import OpenmxScfFW
from atomate.openmx.powerups import add_additional_fields_to_taskdocs, set_execution_options

from pymatgen.io.vasp.inputs import Structure, Kpoints
from fireworks import LaunchPad
from fireworks import Workflow

from monty.serialization import loadfn

from multiprocessing import Pool

import pandas as pd

class GetStructure:
    def __init__(self, files):
        self.df = pd.concat([self.read_json_file(file) for file in files], ignore_index=True)

    @staticmethod
    def read_json_file(file_path):
        return pd.read_json(file_path)
    
    def get_sts(self):
        return [(row['material_id'], Structure.from_dict(row['structure'])) for _, row in self.df.iterrows()]
    
    
class BatchRunner:
    def __init__(self, launchpad_file, st_file):
        self.lpad = LaunchPad.from_file(launchpad_file)
        self.getsts = GetStructure(st_file)
        self.total_sts = len(self.getsts.df)


    def calc(self, batch_id, kppa=2e3):
        st_dict = self.getsts.get_sts()
        print(f"Batch {batch_id} started, Length of st_dict: {len(st_dict)}")
        for i in range((batch_id-1)*10, batch_id*10):
            print(f"Processing structure {i}")
            if i >= len(st_dict):
                break
            mpid = st_dict[i][0]
            st = st_dict[i][1]

            fw = OpenmxScfFW(structure=st, run_deeph_preprocess=True, override_default_openmx_params={"kppa": kppa})
            wf = Workflow([fw], name=f"{mpid}")

            def get_kpts(kppa, structure):
                kpoints = Kpoints.automatic_density(structure, kppa)
                # if kpt in one direction is even, make it odd
                kpoints.kpts[0][0] = kpoints.kpts[0][0] + 1 if kpoints.kpts[0][0] % 2 == 0 else kpoints.kpts[0][0]
                kpoints.kpts[0][1] = kpoints.kpts[0][1] + 1 if kpoints.kpts[0][1] % 2 == 0 else kpoints.kpts[0][1]
                kpoints.kpts[0][2] = kpoints.kpts[0][2] + 1 if kpoints.kpts[0][2] % 2 == 0 else kpoints.kpts[0][2]
                print(kpoints.kpts)
                return kpoints

            print(get_kpts(kppa, st).kpts[0][0])

            wf = add_additional_fields_to_taskdocs(wf, {"mp-id": mpid})
            
            wf = set_execution_options(wf, category="05142024")

            self.lpad.add_wf(wf)

    def missing_calcs(self, kppa=2e3):
        st_dict = loadfn('/workspaces/openmx-wf/SrGeSi/missing_calcs.json')
        for k, _ in st_dict.items():
            print(f"Processing uid {k}")
            st_ase_dict = st_dict[k]
            st = st_ase_dict['st_vasp_dict']
            # remove the st_vasp_dict key 
            st_ase_dict.pop('st_vasp_dict')
            st_ase_dict['file_from'] = '20230805-training_data/initial-train.xyz'

            fw = OpenmxScfFW(structure=st, run_deeph_preprocess=True, override_default_openmx_params={"kppa": kppa})
            wf = Workflow([fw], name=f"{k}")

            def get_kpts(kppa, structure):
                kpoints = Kpoints.automatic_density(structure, kppa)
                # if kpt in one direction is even, make it odd
                kpoints.kpts[0][0] = kpoints.kpts[0][0] + 1 if kpoints.kpts[0][0] % 2 == 0 else kpoints.kpts[0][0]
                kpoints.kpts[0][1] = kpoints.kpts[0][1] + 1 if kpoints.kpts[0][1] % 2 == 0 else kpoints.kpts[0][1]
                kpoints.kpts[0][2] = kpoints.kpts[0][2] + 1 if kpoints.kpts[0][2] % 2 == 0 else kpoints.kpts[0][2]
                print(kpoints.kpts)
                return kpoints

            print(get_kpts(kppa, st).kpts[0][0])

            wf = add_additional_fields_to_taskdocs(wf, {"prev_vasp_calc": st_ase_dict, "uid": k})
            
            wf = set_execution_options(wf, category="20230805-training_data")

            self.lpad.add_wf(wf)        

    def run_batch(self):
        # use parallel to run the function of calc1 from batch 73 to finish
        with Pool(1) as p:
            p.map(self.calc, range(1, int(self.total_sts/10)+1))
        # clean up the memory
        p.close()
        p.join()
        print("Batch finished")



if __name__ == "__main__":
    runner = BatchRunner(
        "/global/homes/a/angush/atomate_test/openmx-wf/Atomate/setting/my_launchpad.yaml", 
        [
            '/global/homes/a/angush/atomate_test/openmx-wf/materials_project/05142024/mp_structure_4000-4500.json',
            '/global/homes/a/angush/atomate_test/openmx-wf/materials_project/05142024/mp_structure_4500-5000.json',
            '/global/homes/a/angush/atomate_test/openmx-wf/materials_project/05142024/mp_structure_5000-5500.json',
        ]
    )
    # runner.missing_calcs()
    runner.run_batch()

    # getsts = GetStructure(files)
    # print(getsts.df)

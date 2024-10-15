"""
Fields must be in the https://gitlab.com/ase/ase/-/blob/master/ase/calculators/openmx/parameters.py?ref_type=heads
"""


import os
from monty.serialization import loadfn, dumpfn
from pymatgen.io.vasp.inputs import Structure, Kpoints

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PAO_TABLE = loadfn(os.path.join(MODULE_DIR, "potential_table.yaml"))

class BaseFileWriter:
    def __init__(self, template):
        self.template = template

    def get_yaml(self):
        dumpfn(self.template, "temp.yaml")

class System(BaseFileWriter):
    def __init__(self, system_current_dir=".", level_of_stdout=1, level_of_fileout=1, **system_settings):
        template = {
            "system_currentdirectory": system_current_dir,  
            "level_of_stdout": level_of_stdout,
            "level_of_fileout": level_of_fileout,
        }

        # allow template to be updated with kwargs
        template.update(system_settings)

        super().__init__(template)

class Species(BaseFileWriter):
    def __init__(self, species_definition):
        template = {
            "definition_of_atomic_species": species_definition
        }
        
        super().__init__(template)

    @classmethod
    def get_species_from_vps_and_option(cls, vpss_and_options):
        output = []
        for vps, option in vpss_and_options.items():
            # if option is not in the list of ["Quick", "Standard", "Precise"], raise ValueError
            if option not in ["Quick", "Standard", "Precise"]:
                raise ValueError(f"Option {option} not in list of options")
            
            # Find the dictionary with the matching VPS
            for d in PAO_TABLE:
                if d['VPS'] == vps:
                    element = vps.split('_')[0]
                    pao_string = f"{element} {d[option]} {vps}"
                    output.append(pao_string.split(" "))
                    break
            else:
                # If we get here, we didn't find a match
                raise ValueError(f"VPS {vps} not found in PAO-table.yaml")

        # call __init__ to get the template
        return Species(species_definition=output)




class Scf(BaseFileWriter):
    def __init__(
        self, 
        kgrid, 
        xc_type="GGA-PBE",
        spin_polarization="off", 
        eigenvalue_solver="Band",
        mixing_type="Simple",
        spin_orbit_coupling="off",
        electronic_temperature=300,
        energy_cutoff=200,
        max_iter=40,
        init_mixing_weight=0.30,
        min_mixing_weight=0.001,
        max_mixing_weight=0.40,
        mixing_history=5,
        start_pulay=6,
        every_pulay=6,
        criterion=1.0e-6,
        **scf_settings
    ):
        template = {
            'scf_xctype': xc_type,
            'scf_spinpolarization': spin_polarization,
            'scf_spinorbit_coupling': spin_orbit_coupling,
            'scf_electronictemperature': electronic_temperature,
            'scf_energycutoff': energy_cutoff,
            'scf_maxiter': max_iter,
            'scf_eigenvaluesolver': eigenvalue_solver,
            'scf_kgrid': kgrid,
            'scf_mixing_type': mixing_type,
            'scf_init_mixing_weight': init_mixing_weight,
            'scf_min_mixing_weight': min_mixing_weight,
            'scf_max_mixing_weight': max_mixing_weight,
            'scf_mixing_history': mixing_history,
            'scf_mixing_startpulay': start_pulay,
            'scf_mixing_everypulay': every_pulay,
            'scf_criterion': criterion,
        }

        # allow template to be updated with kwargs
        template.update(scf_settings)

        super().__init__(template)

    @classmethod
    def get_kgrid_from_pmg_structure(cls, structure, kppa, force_gamma=False):
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma)
        kgrid = kpoints.as_dict()["kpoints"][0]
        kgrid = (kgrid[0], kgrid[1], kgrid[2])
        # if any in kgrid is not even, add 1 to it
        for i in range(3):
            if kgrid[i] % 2 != 0:
                kgrid[i] += 1
        return kgrid
    
    @classmethod
    # initialize the class with the kgrid from get_kgrid_from_pmg_structure
    def get_scf_with_pmg_kgrid(cls, structure, kppa=64, force_gamma=False, **kwargs):
        kgrid = cls.get_kgrid_from_pmg_structure(structure, kppa, force_gamma)
        return cls(kgrid, **kwargs)
    


class MD(BaseFileWriter):
    def __init__(self, md_type, md_max_iter=1, md_time_step=0.5, md_opt_criterion=1.0e-4, **md_settings):
        self.md_type = md_type 
        self.md_max_iter = md_max_iter
        self.md_time_step = md_time_step
        self.md_opt_criterion = md_opt_criterion

        template = {
            "md_type": md_type,
            "md_maxiter": md_max_iter,
            "md_timestep": md_time_step,
            "md_opt_criterion": md_opt_criterion
        }

        # allow template to be updated with kwargs
        template.update(md_settings)
        
        super().__init__(template)





if __name__ == "__main__":
    system = System(system_current_dir=".", system_name="test", level_of_stdout=1, level_of_fileout=1)
    print(system.template)

    # #test get_species_from_vps_and_option
    vpss_and_options = {"Ga_PBE19": "Quick", "As_PBE19": "Quick"}
    species = Species.get_species_from_vps_and_option(vpss_and_options)
    print(species.template)

    # # Test get_scf_with_pmg_kgrid
    structure = Structure.from_file(f"{MODULE_DIR}/POSCAR")
    scf = Scf.get_scf_with_pmg_kgrid(structure, kppa=64, force_gamma=False)
    print(scf.template)


    md = MD(md_type="nomd", md_max_iter=1, md_time_step=0.5, md_opt_criterion=1.0e-4)
    print(md.template)


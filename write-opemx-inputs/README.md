# How this `firetask.py` works for creating OpenMX input file

## Parameters to write openmx.dat
The fundamental parameters are stored in `/src/pymatgen/pymatgen/io/openmx/IncarInputSet.yaml`.

## KPOINTS parameters
- Check parameter `scf/kppa` in `/src/pymatgen/pymatgen/io/openmx/KpointsInputSet.yaml`.
- Following function  is used to generate KPOINTS file (in `/src/pymatgen/pymatgen/io/openmx/inputs.py`)
```python
    def get_kgrid_from_pmg_structure(cls, structure, kppa, force_gamma=False):
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma)
        kgrid = kpoints.as_dict()["kpoints"][0]
        kgrid = (kgrid[0], kgrid[1], kgrid[2])
        # if any in kgrid is not even, add 1 to it
        for i in range(3):
            print(f"{i}: {kgrid[i]}")
            if kgrid[i] % 2 == 0:
                print("even, adding 1")
                new_k = kgrid[i] + 1
                kgrid = list(kgrid)
                kgrid[i] = new_k
                kgrid = tuple(kgrid)
        return kgrid
```

## POTCAR table
- Check the file `/src/pymatgen/pymatgen/io/openmx/potential_table.yaml` for the list of potentials available.


## How to run the script
- The script is run by `python write_openmx_inputs.py` in the terminal. There is a `/openmx/GaAs.vasp` file that is used as input to generate the OpenMX input files as an example.
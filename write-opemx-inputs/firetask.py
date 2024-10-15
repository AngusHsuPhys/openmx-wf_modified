from atomate.openmx.firetasks.write_inputs import WriteOpenmxFromIOSet

from pymatgen.io.vasp.inputs import Structure



def write_openmx_inputs(structure, openmx_input_set, openmx_dft_data_path, override_default_openmx_params, potcar_spec, magmoms):
    write_openmx_inputs = WriteOpenmxFromIOSet(
        structure=structure,
        openmx_input_set=openmx_input_set,
        openmx_dft_data_path=openmx_dft_data_path,
        openmx_input_params=override_default_openmx_params,
        potcar_spec=potcar_spec,
        magmoms=magmoms,
    )

    write_openmx_inputs.run_task({})


if __name__ == "__main__":
    structure = Structure.from_file("/openmx/GaAs.vasp")
    openmx_input_set = "ScfInputSet"
    openmx_dft_data_path = "/openmx/DFT_DATA19"

    override_default_openmx_params = {"kppa": 2e3}
    potcar_spec = None
    magmoms = None

    write_openmx_inputs(structure, openmx_input_set, openmx_dft_data_path, override_default_openmx_params, potcar_spec, magmoms)

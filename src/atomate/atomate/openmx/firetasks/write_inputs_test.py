"""
This module defines tasks for writing vasp input sets.
"""

import os
from importlib import import_module

import numpy as np
from fireworks import FiretaskBase, explicit_serialize
from fireworks.utilities.dict_mods import apply_mod
from monty.serialization import dumpfn
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar, PotcarSingle
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import (
    MPHSEBSSet,
    MPNMRSet,
    MPNonSCFSet,
    MPScanRelaxSet,
    MPSOCSet,
    MPStaticSet,
)

from atomate.utils.utils import env_chk, load_class
from atomate.vasp.firetasks.glue_tasks import GetInterpolatedPOSCAR

__author__ = "Anubhav Jain, Shyue Ping Ong, Kiran Mathew, Alex Ganose"
__email__ = "ajain@lbl.gov"


@explicit_serialize
class WriteVaspFromIOSet(FiretaskBase):
    """
    Create VASP input files using implementations of pymatgen's
    AbstractVaspInputSet. An input set can be provided as an object or as a
    String/parameter combo.

    Required params:
        structure (Structure): structure
        vasp_input_set (AbstractVaspInputSet or str): Either a VaspInputSet
            object or a string name for the VASP input set (e.g., "MPRelaxSet").

    Optional params:
        vasp_input_params (dict): When using a string name for VASP input set,
            use this as a dict to specify kwargs for instantiating the input set
            parameters. For example, if you want to change the
            user_incar_settings, you should provide:
            {"user_incar_settings": ...}. This setting is ignored if you provide
            the full object representation of a VaspInputSet rather than a
            String.
        potcar_spec (bool): Instead of writing the POTCAR, write a
            "POTCAR.spec". This is intended to allow testing of workflows
            without requiring pseudo-potentials to be installed on the system.
        spec_structure_key (str): If supplied, then attempt to read this from the fw_spec
            to obtain the structure
    """

    required_params = ["structure", "vasp_input_set"]
    optional_params = ["vasp_input_params", "potcar_spec", "spec_structure_key"]

    def run_task(self, fw_spec):
        # if a full VaspInputSet object was provided
        if hasattr(self["vasp_input_set"], "write_input"):
            vis = self["vasp_input_set"]

        # if VaspInputSet String + parameters was provided
        else:
            vis_cls = load_class("pymatgen.io.vasp.sets", self["vasp_input_set"])
            vis = vis_cls(self["structure"], **self.get("vasp_input_params", {}))

        # over-write structure with fw_spec structure
        spec_structure_key = self.get("spec_structure_key", None)
        if spec_structure_key is not None:
            fw_struct = fw_spec.get(spec_structure_key)
            dd = vis.as_dict()
            dd["structure"] = fw_struct
            vis = vis.from_dict(dd)

        potcar_spec = self.get("potcar_spec", False)
        vis.write_input(".", potcar_spec=potcar_spec)
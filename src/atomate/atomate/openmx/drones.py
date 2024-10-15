"""
This Drone tries to produce a more sensible task dictionary than the default VaspToDbTaskDrone.
Some of the changes are documented in this thread:
https://groups.google.com/forum/#!topic/pymatgen/pQ-emBpeV5U
"""

import datetime
import glob
import json
import os
import re
import traceback
import warnings
from fnmatch import fnmatch
from shutil import which

import numpy as np
from monty.io import zopen
from monty.json import jsanitize
from pymatgen.apps.borg.hive import AbstractDrone
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.core.composition import Composition
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.io.vasp import BSVasprun, Locpot, Outcar, Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate import __version__ as atomate_version
from atomate.utils.utils import get_logger, get_uri
from atomate.vasp.config import (
    STORE_ADDITIONAL_JSON,
    STORE_BADER,
    STORE_VOLUMETRIC_DATA,
)

from ase.calculators.openmx.reader import read_file

__author__ = "Kiran Mathew, Shyue Ping Ong, Shyam Dwaraknath, Anubhav Jain"
__email__ = "kmathew@lbl.gov"
__date__ = "Mar 27, 2016"
__version__ = "0.1.0"

logger = get_logger(__name__)

BADER_EXE_EXISTS = which("bader") or which("bader.exe")
STORE_BADER = STORE_BADER and BADER_EXE_EXISTS


class openmxDrone(AbstractDrone):
    """
    pymatgen-db VaspToDbTaskDrone with updated schema and documents processing methods.
    Please refer to matgendb.creator.VaspToDbTaskDrone documentation.
    """

    __version__ = atomate_version  # note: the version is inserted into the task doc

    # Schema def of important keys and sub-keys; used in validation
    schema = {
        "root": {
            "schema",
            "dir_name",
            "chemsys",
            "composition_reduced",
            "formula_pretty",
            "formula_reduced_abc",
            "elements",
            "nelements",
            "formula_anonymous",
            "calcs_reversed",
            "completed_at",
            "nsites",
            "composition_unit_cell",
            "input",
            "output",
            "state",
            "analysis",
            "run_stats",
        },
        "input": {
            "is_lasph",
            "is_hubbard",
            "xc_override",
            "potcar_spec",
            "hubbards",
            "structure",
            "pseudo_potential",
        },
        "output": {
            "structure",
            "spacegroup",
            "density",
            "energy",
            "energy_per_atom",
            "is_gap_direct",
            "bandgap",
            "vbm",
            "cbm",
            "is_metal",
            "forces",
            "stress",
        },
        "calcs_reversed": {
            "dir_name",
            "run_type",
            "elements",
            "nelements",
            "formula_pretty",
            "formula_reduced_abc",
            "composition_reduced",
            "vasp_version",
            "formula_anonymous",
            "nsites",
            "composition_unit_cell",
            "completed_at",
            "task",
            "input",
            "output",
            "has_vasp_completed",
        },
        "analysis": {
            "delta_volume_as_percent",
            "delta_volume",
            "max_force",
            "errors",
            "warnings",
        },
    }

    def __init__(
        self,
        additional_fields=None,
        use_full_uri=True,
        parse_out=True,
        parse_scfout=True,
        parse_deeph=False,
        store_volumetric_data=STORE_VOLUMETRIC_DATA,
        store_additional_json=STORE_ADDITIONAL_JSON,
        parse_resume=True,
    ):
        """
        Initialize a Vasp drone to parse VASP outputs
        Args:
            runs (list): Naming scheme for multiple calculations in on folder e.g. ["relax1","relax2"].
             Can be subfolder or extension
            parse_dos (str or bool): Whether to parse the DOS. Can be "auto", True or False.
            "auto" will only parse DOS if NSW = 0, so there are no ionic steps
            bandstructure_mode (str or bool): How to parse the bandstructure or not.
            Can be "auto","line", True or False.
             "auto" will parse the bandstructure with projections for NSCF calcs and decide automatically
              if it's line mode or uniform. Saves the bandstructure in the output doc.
             "line" will parse the bandstructure as a line mode calculation with projections.
              Saves the bandstructure in the output doc.
             True will parse the bandstructure with projections as a uniform calculation.
              Saves the bandstructure in the output doc.
             False will parse the bandstructure without projections to calculate vbm, cbm, band_gap, is_metal and efermi
              Dose not saves the bandstructure in the output doc.
            parse_locpot (bool): Parses the LOCPOT file and saves the 3 axis averages
            additional_fields (dict): dictionary of additional fields to add to output document
            use_full_uri (bool): converts the directory path to the full URI path
            parse_bader (bool): Run and parse Bader charge data. Defaults to True if Bader is present
            parse_chgcar (bool): Run and parse CHGCAR file
            parse_aeccar (bool): Run and parse AECCAR0 and AECCAR2 files
            store_volumetric_data (list): List of files to store, choose from ('CHGCAR', 'LOCPOT',
            'AECCAR0', 'AECCAR1', 'AECCAR2', 'ELFCAR'), case insensitive
            store_additional_json (bool): If True, parse any .json files present and store as
            sub-doc including the FW.json if present
        """
        self.additional_fields = additional_fields or {}
        self.use_full_uri = use_full_uri
        self.store_volumetric_data = [f.lower() for f in store_volumetric_data]
        self.store_additional_json = store_additional_json
        self.parse_out = parse_out
        self.parse_scfout = parse_scfout
        self.parse_deeph = parse_deeph
        self.parse_resume = parse_resume

    def assimilate(self, path):
        """
        Adapted from matgendb.creator
        Parses vasp runs(vasprun.xml file) and insert the result into the db.
        Get the entire task doc from the vasprum.xml and the OUTCAR files in the path.
        Also adds some post-processed info.

        Args:
            path (str): Path to the directory containing vasprun.xml and OUTCAR files

        Returns:
            (dict): a task dictionary
        """
        logger.info(f"Getting task doc for base dir :{path}")
        d = self.generate_doc(path)
        return d

    def filter_files(self, path, file_pattern="vasprun.xml"):
        """
        Find the files that match the pattern in the given path and
        return them in an ordered dictionary. The searched for files are
        filtered by the run types defined in self.runs. e.g. ["relax1", "relax2", ...].
        Only 2 schemes of the file filtering is enabled: searching for run types
        in the list of files and in the filenames. Modify this method if more
        sophisticated filtering scheme is needed.

        Args:
            path (string): path to the folder
            file_pattern (string): files to be searched for

        Returns:
            dict: names of the files to be processed further. The key is set from list
                of run types: self.runs
        """
        processed_files = {}
        files = os.listdir(path)
        for r in self.runs:
            # try subfolder schema
            if r in files:
                for f in os.listdir(os.path.join(path, r)):
                    if fnmatch(f, f"{file_pattern}*"):
                        processed_files[r] = os.path.join(r, f)
            # try extension schema
            else:
                for f in files:
                    if fnmatch(f, f"{file_pattern}.{r}*"):
                        processed_files[r] = f
        if len(processed_files) == 0:
            # get any matching file from the folder
            for f in files:
                if fnmatch(f, f"{file_pattern}*"):
                    processed_files["standard"] = f
        return processed_files

    def generate_doc(self, dir_name):
        """
        Adapted from matgendb.creator.generate_doc
        """
        try:
            # basic properties, incl. calcs_reversed and run_stats
            fullpath = os.path.abspath(dir_name)
            d = jsanitize(self.additional_fields, strict=True)
            d["dir_name"] = fullpath
            st = Structure.from_file(os.path.join(fullpath, "POSCAR"))
            d["pmg_structure"] = st.as_dict()
            d["formula"] = st.formula
            openmx_out_file = os.path.join(fullpath, "openmx.out")

            try:
                d["ase_calc"] = read_file(openmx_out_file)
                d["state"] = "successful"
            except Exception:
                logger.error(f"Error reading {openmx_out_file}")
                raise

            d["calcs_reversed"] = [{}]
            for file_name in ["openmx.out", "openmx.scfout"]:
                if getattr(self, f'parse_{file_name.split(".")[-1]}'):
                    # d["calcs_reversed"][0].update(self.process_out(dir_name, file_name))
                    # "openmx_raw" can be not exist in d["calcs_reversed"][0], make sure it is updated
                    if "openmx_raw" in d["calcs_reversed"][0]:
                        d["calcs_reversed"][0]["openmx_raw"].update(self.process_out(fullpath, file_name))
                    else:
                        d["calcs_reversed"][0]["openmx_raw"] = self.process_out(fullpath, file_name)


            # if self.parse_deeph is true, parse the deeph file
            if self.parse_deeph:
                deeph_base_dir = os.path.join(fullpath, "deeph", "processed_dir")

                # check if deeph_base_dir exists and does not contain error.log file
                if os.path.exists(deeph_base_dir) and not os.path.exists(os.path.join(deeph_base_dir, "error.log")):
                    # read info.json file and convert to dict
                    with open(os.path.join(deeph_base_dir, "info.json"), "r") as f:
                        d["deeph"] = json.load(f)
                    #scan the deeph_base_dir for deeph files and update the calcs_reversed with the output of process_out
                    for file_name in os.listdir(deeph_base_dir):
                        # escape info.json file
                        if file_name != "info.json":
                            # d["calcs_reversed"][0].update(self.process_out(deeph_base_dir, file_name))
                            if "deeph_raw" in d["calcs_reversed"][0]:
                                d["calcs_reversed"][0]["deeph_raw"].update(self.process_out(deeph_base_dir, file_name))
                            else:
                                d["calcs_reversed"][0]["deeph_raw"] = self.process_out(deeph_base_dir, file_name)
                else:
                    logger.error(f"deeph_base_dir {deeph_base_dir} does not exist")
                    raise ValueError(f"deeph_base_dir {deeph_base_dir} does not exist")


            if self.parse_resume:
                rst_dir = os.path.join(fullpath, "openmx_rst")
                # check if the folder exists
                if os.path.exists(rst_dir):
                    for file_name in os.listdir(rst_dir):
                        # d["calcs_reversed"][0].update(self.process_out(rst_dir, file_name))
                        if "openmx_rst" in d["calcs_reversed"][0]:
                            d["calcs_reversed"][0]["openmx_rst"].update(self.process_out(rst_dir, file_name))
                        else:
                            d["calcs_reversed"][0]["openmx_rst"] = self.process_out(rst_dir, file_name)
                else:
                    logger.error(f"rst_dir {rst_dir} does not exist")
                    raise ValueError(f"rst_dir {rst_dir} does not exist")


            d["last_updated"] = datetime.datetime.utcnow()
            return d

        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                "Error in " + os.path.abspath(dir_name) + ".\n" + traceback.format_exc()
            )
            raise

    def process_out(self, dir_name, filename):
        f = os.path.join(dir_name, filename)
        # if filename contains ".out" then it is a standard file, then use "'ISO-8859-1" encoding
        # if ".out" in filename:
        #     with open(f, "r", encoding="ISO-8859-1") as f:
        #         data = f.read()
        # if filename container ".scfout" then it is a binary file. Convert to something that can be dumped to json
        # elif ".scfout" in filename:
        data = f
        d = {}
        # replace dots in filename with underscores to avoid mongo issues
        d[f"{filename.replace('.', '_')}"] = data
        return d

    def process_vasprun(self, dir_name, taskname, filename):
        """
        Adapted from matgendb.creator

        Process a vasprun.xml file.
        """
        vasprun_file = os.path.join(dir_name, filename)

        vrun = Vasprun(vasprun_file, parse_potcar_file=self.parse_potcar_file)

        d = vrun.as_dict()

        # rename formula keys
        for k, v in {
            "formula_pretty": "pretty_formula",
            "composition_reduced": "reduced_cell_formula",
            "composition_unit_cell": "unit_cell_formula",
        }.items():
            d[k] = d.pop(v)

        for k in [
            "eigenvalues",
            "projected_eigenvalues",
        ]:  # large storage space breaks some docs
            if k in d["output"]:
                del d["output"][k]

        comp = Composition(d["composition_unit_cell"])
        d["formula_anonymous"] = comp.anonymized_formula
        d["formula_reduced_abc"] = comp.reduced_composition.alphabetical_formula
        d["dir_name"] = os.path.abspath(dir_name)
        d["completed_at"] = str(
            datetime.datetime.fromtimestamp(os.path.getmtime(vasprun_file))
        )
        d["density"] = vrun.final_structure.density

        # replace 'crystal' with 'structure'
        d["input"]["structure"] = d["input"].pop("crystal")
        d["output"]["structure"] = d["output"].pop("crystal")
        for k, v in {
            "energy": "final_energy",
            "energy_per_atom": "final_energy_per_atom",
        }.items():
            d["output"][k] = d["output"].pop(v)

        # Process bandstructure and DOS
        if self.bandstructure_mode is not False:  # noqa
            bs = self.process_bandstructure(vrun)
            if bs:
                d["bandstructure"] = bs

        if self.parse_dos is not False:  # noqa
            dos = self.process_dos(vrun)
            if dos:
                d["dos"] = dos

        # Parse electronic information if possible.
        # For certain optimizers this is broken and we don't get an efermi resulting in the bandstructure
        try:
            bs = vrun.get_band_structure(efermi="smart")
            bs_gap = bs.get_band_gap()
            d["output"]["vbm"] = bs.get_vbm()["energy"]
            d["output"]["cbm"] = bs.get_cbm()["energy"]
            d["output"]["bandgap"] = bs_gap["energy"]
            d["output"]["is_gap_direct"] = bs_gap["direct"]
            d["output"]["is_metal"] = bs.is_metal()
            if not bs_gap["direct"]:
                d["output"]["direct_gap"] = bs.get_direct_band_gap()
            if isinstance(bs, BandStructureSymmLine):
                d["output"]["transition"] = bs_gap["transition"]

        except Exception:
            logger.warning("Error in parsing bandstructure")
            if vrun.incar["IBRION"] == 1:
                logger.warning("Vasp doesn't properly output efermi for IBRION == 1")
            if self.bandstructure_mode is True:
                logger.error(traceback.format_exc())
                logger.error(
                    "Error in "
                    + os.path.abspath(dir_name)
                    + ".\n"
                    + traceback.format_exc()
                )
                raise

        # Should roughly agree with information from .get_band_structure() above, subject to tolerances
        # If there is disagreement, it may be related to VASP incorrectly assigning the Fermi level
        try:
            band_props = vrun.eigenvalue_band_properties
            d["output"]["eigenvalue_band_properties"] = {
                "bandgap": band_props[0],
                "cbm": band_props[1],
                "vbm": band_props[2],
                "is_gap_direct": band_props[3],
            }
        except Exception:
            logger.warning("Error in parsing eigenvalue band properties")

        # store run name and location ,e.g. relax1, relax2, etc.
        d["task"] = {"type": taskname, "name": taskname}

        # include output file names
        d["output_file_paths"] = self.process_raw_data(dir_name, taskname=taskname)

        # parse axially averaged locpot
        if "locpot" in d["output_file_paths"] and self.parse_locpot:
            locpot = Locpot.from_file(
                os.path.join(dir_name, d["output_file_paths"]["locpot"])
            )
            d["output"]["locpot"] = {
                i: locpot.get_average_along_axis(i) for i in range(3)
            }

        if self.store_volumetric_data:
            for file in self.store_volumetric_data:
                if file in d["output_file_paths"]:
                    try:
                        # assume volumetric data is all in CHGCAR format
                        data = Chgcar.from_file(
                            os.path.join(dir_name, d["output_file_paths"][file])
                        )
                        d[file] = data.as_dict()
                    except Exception:
                        raise ValueError(
                            f"Failed to parse {file} at {d['output_file_paths'][file]}."
                        )

        # parse force constants
        if hasattr(vrun, "force_constants"):
            d["output"]["force_constants"] = vrun.force_constants.tolist()
            d["output"]["normalmode_eigenvals"] = vrun.normalmode_eigenvals.tolist()
            d["output"]["normalmode_eigenvecs"] = vrun.normalmode_eigenvecs.tolist()

        # perform Bader analysis using Henkelman bader
        if self.parse_bader and "chgcar" in d["output_file_paths"]:
            suffix = "" if taskname == "standard" else f".{taskname}"
            bader = bader_analysis_from_path(dir_name, suffix=suffix)
            d["bader"] = bader

        # parse output from loptics
        if vrun.incar.get("LOPTICS", False):
            dielectric = vrun.dielectric
            d["output"]["dielectric"] = dict(
                energy=dielectric[0], real=dielectric[1], imag=dielectric[2]
            )
            d["output"]["optical_absorption_coeff"] = vrun.optical_absorption_coeff

        # parse output from response function
        if vrun.incar.get("ALGO") == "CHI":
            dielectric = vrun.dielectric
            d["output"]["dielectric"] = dict(
                energy=dielectric[0], real=dielectric[1], imag=dielectric[2]
            )
            d["output"]["optical_absorption_coeff"] = vrun.optical_absorption_coeff

        return d

    def process_bandstructure(self, vrun):

        vasprun_file = vrun.filename
        # Band structure parsing logic
        if str(self.bandstructure_mode).lower() == "auto":
            # if NSCF calculation
            if vrun.incar.get("ICHARG", 0) > 10:
                bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
                try:
                    # Try parsing line mode
                    bs = bs_vrun.get_band_structure(line_mode=True)
                except Exception:
                    # Just treat as a regular calculation
                    bs = bs_vrun.get_band_structure()
            # else just regular calculation
            else:
                bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=False)
                bs = bs_vrun.get_band_structure()

            # only save the bandstructure if not moving ions
            if vrun.incar.get("NSW", 0) <= 1:
                return bs.as_dict()

        # legacy line/True behavior for bandstructure_mode
        elif self.bandstructure_mode:
            bs_vrun = BSVasprun(vasprun_file, parse_projected_eigen=True)
            bs = bs_vrun.get_band_structure(
                line_mode=(str(self.bandstructure_mode).lower() == "line")
            )
            return bs.as_dict()

        return None

    def process_dos(self, vrun):
        # parse dos if forced to or auto mode set and  0 ionic steps were performed -> static calculation and not DFPT
        if self.parse_dos is True or (
            str(self.parse_dos).lower() == "auto" and vrun.incar.get("NSW", 0) < 1
        ):
            try:
                return vrun.complete_dos.as_dict()
            except Exception:
                raise ValueError("No valid dos data exist")

    def process_raw_data(self, dir_name, taskname="standard"):
        """
        It is useful to store what raw data has been calculated
        and exists for easier querying of the taskdoc.

        :param dir_name: directory to search
        :param taskname: taskname, e.g. "relax1"
        :return: dict of files present
        """
        d = {}
        possible_files = (
            "openmx.out",
            "openmx.scfout"
        )
        for f in possible_files:
            files = self.filter_files(dir_name, file_pattern=f)
            if taskname in files:
                d[f.lower()] = files[taskname]
        return d

    @staticmethod
    def set_analysis(d, max_force_threshold=0.5, volume_change_threshold=0.2):
        """
        Adapted from matgendb.creator

        set the 'analysis' key
        """
        initial_vol = d["input"]["structure"]["lattice"]["volume"]
        final_vol = d["output"]["structure"]["lattice"]["volume"]
        delta_vol = final_vol - initial_vol
        percent_delta_vol = 100 * delta_vol / initial_vol
        warning_msgs = []
        error_msgs = []

        # delta volume checks
        if abs(percent_delta_vol) > volume_change_threshold:
            warning_msgs.append(f"Volume change > {volume_change_threshold * 100}%")

        # max force and valid structure checks
        max_force = None
        calc = d["calcs_reversed"][0]
        if d["state"] == "successful":

            # calculate max forces
            if "forces" in calc["output"]["ionic_steps"][-1]:
                forces = np.array(calc["output"]["ionic_steps"][-1]["forces"])
                # account for selective dynamics
                final_structure = Structure.from_dict(calc["output"]["structure"])
                sdyn = final_structure.site_properties.get("selective_dynamics")
                if sdyn:
                    forces[np.logical_not(sdyn)] = 0
                max_force = max(np.linalg.norm(forces, axis=1))

            if calc["input"]["parameters"].get("NSW", 0) > 0:

                drift = calc["output"]["outcar"].get("drift", [[0, 0, 0]])
                max_drift = max(np.linalg.norm(d) for d in drift)
                ediffg = calc["input"]["parameters"].get("EDIFFG", None)
                if ediffg and float(ediffg) < 0:
                    desired_force_convergence = -float(ediffg)
                else:
                    desired_force_convergence = np.inf
                if max_drift > desired_force_convergence:
                    warning_msgs.append(
                        f"Drift ({drift}) > desired force convergence ({desired_force_convergence}), "
                        "structure likely not converged to desired accuracy."
                    )

            s = Structure.from_dict(d["output"]["structure"])
            if not s.is_valid():
                error_msgs.append("Bad structure (atoms are too close!)")
                d["state"] = "error"

        d["analysis"] = {
            "delta_volume": delta_vol,
            "delta_volume_as_percent": percent_delta_vol,
            "max_force": max_force,
            "warnings": warning_msgs,
            "errors": error_msgs,
        }

    def post_process(self, dir_name, d):
        """
        Post-processing for various files other than the vasprun.xml and OUTCAR.
        Looks for files: transformations.json and custodian.json. Modify this if other
        output files need to be processed.

        Args:
            dir_name:
                The dir_name.
            d:
                Current doc generated.
        """
        logger.info(f"Post-processing dir:{dir_name}")
        fullpath = os.path.abspath(dir_name)
        # VASP input generated by pymatgen's alchemy has a transformations.json file that tracks
        # the origin of a particular structure. If such a file is found, it is inserted into the
        # task doc as d["transformations"]
        transformations = {}
        filenames = glob.glob(os.path.join(fullpath, "transformations.json*"))
        if len(filenames) >= 1:
            with zopen(filenames[0], "rt") as f:
                transformations = json.load(f)
                try:
                    m = re.match(r"(\d+)-ICSD", transformations["history"][0]["source"])
                    if m:
                        d["icsd_id"] = int(m.group(1))
                except Exception:
                    logger.warning("Cannot parse ICSD from transformations file.")
        else:
            logger.warning("Transformations file does not exist.")

        other_parameters = transformations.get("other_parameters")
        new_tags = None
        if other_parameters:
            # We don't want to leave tags or authors in the
            # transformations file because they'd be copied into
            # every structure generated after this one.
            new_tags = other_parameters.pop("tags", None)
            new_author = other_parameters.pop("author", None)
            if new_author:
                d["author"] = new_author
            if not other_parameters:  # if dict is now empty remove it
                transformations.pop("other_parameters")
        d["transformations"] = transformations

        # Calculations done using custodian has a custodian.json,
        # which tracks the jobs performed and any errors detected and fixed.
        # This is useful for tracking what has actually be done to get a
        # result. If such a file is found, it is inserted into the task doc
        # as d["custodian"]
        filenames = glob.glob(os.path.join(fullpath, "custodian.json*"))
        if len(filenames) >= 1:
            custodian = []
            for fname in filenames:
                with zopen(fname, "rt") as f:
                    custodian.append(json.load(f)[0])
            d["custodian"] = custodian
        # Convert to full uri path.
        if self.use_full_uri:
            d["dir_name"] = get_uri(dir_name)
        if new_tags:
            d["tags"] = new_tags

        # Calculations using custodian generate a *.orig file for the inputs
        # This is useful to know how the calculation originally started
        # if such files are found they are inserted into orig_inputs
        filenames = glob.glob(os.path.join(fullpath, "*.orig*"))

        if len(filenames) >= 1:
            d["orig_inputs"] = {}
            for f in filenames:
                if "INCAR.orig" in f:
                    d["orig_inputs"]["incar"] = Incar.from_file(f).as_dict()
                if "POTCAR.orig" in f:
                    d["orig_inputs"]["potcar"] = Potcar.from_file(f).as_dict()
                if "KPOINTS.orig" in f:
                    d["orig_inputs"]["kpoints"] = Kpoints.from_file(f).as_dict()
                if "POSCAR.orig" in f:
                    d["orig_inputs"]["poscar"] = Poscar.from_file(f).as_dict()

        filenames = glob.glob(os.path.join(fullpath, "*.json*"))
        if self.store_additional_json and filenames:
            for filename in filenames:
                key = os.path.basename(filename).split(".")[0]
                if key != "custodian" and key != "transformations":
                    with zopen(filename, "rt") as f:
                        d[key] = json.load(f)

        logger.info("Post-processed " + fullpath)

    def validate_doc(self, d):
        """
        Sanity check.
        Make sure all the important keys are set
        """
        # TODO: @matk86 - I like the validation but I think no one will notice a failed
        # validation tests which removes the usefulness of this. Any ideas to make people
        # notice if the validation fails? -computron
        for k, v in self.schema.items():
            if k == "calcs_reversed":
                diff = v - set(d.get(k, d)[0].keys())
            else:
                diff = v - set(d.get(k, d).keys())
            if diff:
                logger.warning(f"The keys {diff} in {k} not set")

    def get_valid_paths(self, path):
        """
        There are some restrictions on the valid directory structures:

        1. There can be only one vasp run in each directory. Nested directories
           are fine.
        2. Directories designated "relax1"..."relax9" are considered to be
           parts of a multiple-optimization run.
        3. Directories containing vasp output with ".relax1"...".relax9" are
           also considered as parts of a multiple-optimization run.
        """
        (parent, subdirs, files) = path
        if set(self.runs).intersection(subdirs):
            return [parent]
        if (
            not any([parent.endswith(os.sep + r) for r in self.runs])
            and len(glob.glob(os.path.join(parent, "vasprun.xml*"))) > 0
        ):
            return [parent]
        return []

    def as_dict(self):
        init_args = {
            "parse_dos": self.parse_dos,
            "bandstructure_mode": self.bandstructure_mode,
            "additional_fields": self.additional_fields,
            "use_full_uri": self.use_full_uri,
            "runs": self.runs,
        }
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "version": self.__class__.__version__,
            "init_args": init_args,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d["init_args"])

"""
This module defines tasks that support running vasp in various ways.
"""

import os
import shlex
import shutil
import subprocess

from custodian import Custodian
from custodian.vasp.handlers import (
    AliasingErrorHandler,
    DriftErrorHandler,
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    PotimErrorHandler,
    ScanMetalHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
    WalltimeHandler,
)
from custodian.vasp.jobs import VaspJob, VaspNEBJob
from custodian.vasp.validators import VaspFilesValidator, VasprunXMLValidator
from fireworks import FiretaskBase, FWAction, explicit_serialize
from monty.os.path import zpath
from monty.serialization import loadfn
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import get_vasprun_outcar

from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.config import CUSTODIAN_MAX_ERRORS, HALF_KPOINTS_FIRST_RELAX

__author__ = "Anubhav Jain <ajain@lbl.gov>"
__credits__ = "Shyue Ping Ong <ong.sp>"

logger = get_logger(__name__)


@explicit_serialize
class RunOpenmx(FiretaskBase):
    """
    Execute a command directly (no custodian).

    Required params:
        cmd (str): the name of the full executable to run. Supports env_chk.
    Optional params:
        expand_vars (str): Set to true to expand variable names in the cmd.
    """

    required_params = ["openmx_cmd"]
    # optional_params = ["openmx_input_file", "openmx_output_file"]

    def run_task(self, fw_spec):
        cmd = env_chk(self["openmx_cmd"], fw_spec)
        cmd += f" openmx.dat > stat"

        logger.info(f"Running command: {cmd}")
        return_code = subprocess.call(cmd, shell=True)
        logger.info(f"Command {cmd} finished running with returncode: {return_code}")

        # set the state by checking the output file "stat" and check if "The calculation was normally finished." is in it.
        with open("stat", "r") as f:
            if "The calculation was normally finished." in f.read():
                print("The calculation was normally finished.")
                return FWAction(stored_data={"state": "successful"})
            else:
                raise RuntimeError(f"State of the calculation is not successful. Please check the output file 'stat' for more information.")



@explicit_serialize
class RunDeephPreprocess(FiretaskBase):
    """
    Execute a command directly (no custodian).

    Required params:
        cmd (str): the name of the full executable to run. Supports env_chk.
    Optional params:
        expand_vars (str): Set to true to expand variable names in the cmd.
    """

    required_params = ["deeph_preprocess_cmd"]
    # optional_params = ["openmx_input_file", "openmx_output_file"]

    def run_task(self, fw_spec):
        cmd = env_chk(self["deeph_preprocess_cmd"], fw_spec)
        # cmd += f" main `pwd`"

        logger.info(f"Running command: {cmd}")
        return_code = subprocess.call(cmd, shell=True)
        logger.info(f"Command {cmd} finished running with returncode: {return_code}")
        if return_code != 0:
            raise RuntimeError(f"Deeph Preprocess returned non-zero exit status: {return_code}")

@explicit_serialize
class RunVaspCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", i.e. in a simple way that supports most common options.

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP. Supports env_chk.

    Optional params:
        job_type: (str) - choose from "normal" (default), "double_relaxation_run" (two consecutive
            jobs), "full_opt_run" (multiple optimizations), and "neb"
        handler_group: (str | list[ErrorHandler]) - group of handlers to use. See handler_groups dict in the code or
            the groups and complete list of handlers in each group. Alternatively, you can
            specify a list of ErrorHandler objects.
        scratch_dir: (str) - if specified, uses this directory as the root scratch dir.
            Supports env_chk.
        gzip_output: (bool) - gzip output (default=T)
        max_errors: (int) - maximum # of errors to fix before giving up (default=5)
        ediffg: (float) shortcut for setting EDIFFG in special custodian jobs
        auto_npar: (bool) - use auto_npar (default=F). Recommended set to T
            for single-node jobs only. Supports env_chk.
        gamma_vasp_cmd: (str) - cmd for Gamma-optimized VASP compilation.
            Supports env_chk.
        wall_time (int): Total wall time in seconds. Activates WalltimeHandler if set.
        half_kpts_first_relax (bool): Use half the k-points for the first relaxation
    """

    required_params = ["vasp_cmd"]

    def run_task(self, fw_spec):

        handler_groups = {
            "default": [
                VaspErrorHandler(),
                MeshSymmetryErrorHandler(),
                UnconvergedErrorHandler(),
                NonConvergingErrorHandler(),
                PotimErrorHandler(),
                PositiveEnergyErrorHandler(),
                FrozenJobErrorHandler(),
                StdErrHandler(),
                LargeSigmaHandler(),
                IncorrectSmearingHandler(),
            ],
            "strict": [
                VaspErrorHandler(),
                MeshSymmetryErrorHandler(),
                UnconvergedErrorHandler(),
                NonConvergingErrorHandler(),
                PotimErrorHandler(),
                PositiveEnergyErrorHandler(),
                FrozenJobErrorHandler(),
                StdErrHandler(),
                AliasingErrorHandler(),
                DriftErrorHandler(),
                LargeSigmaHandler(),
                IncorrectSmearingHandler(),
            ],
            "scan": [
                VaspErrorHandler(),
                MeshSymmetryErrorHandler(),
                UnconvergedErrorHandler(),
                NonConvergingErrorHandler(),
                PotimErrorHandler(),
                PositiveEnergyErrorHandler(),
                FrozenJobErrorHandler(),
                StdErrHandler(),
                LargeSigmaHandler(),
                IncorrectSmearingHandler(),
                ScanMetalHandler(),
            ],
            "md": [VaspErrorHandler(), NonConvergingErrorHandler()],
            "no_handler": [],
        }

        vasp_cmd = env_chk(self["vasp_cmd"], fw_spec)

        if isinstance(vasp_cmd, str):
            vasp_cmd = os.path.expandvars(vasp_cmd)
            vasp_cmd = shlex.split(vasp_cmd)

        # initialize variables
        job_type = self.get("job_type", "normal")
        scratch_dir = env_chk(self.get("scratch_dir"), fw_spec)
        gzip_output = self.get("gzip_output", True)
        max_errors = self.get("max_errors", CUSTODIAN_MAX_ERRORS)
        auto_npar = env_chk(self.get("auto_npar"), fw_spec, strict=False, default=False)
        gamma_vasp_cmd = env_chk(
            self.get("gamma_vasp_cmd"), fw_spec, strict=False, default=None
        )
        if gamma_vasp_cmd:
            gamma_vasp_cmd = shlex.split(gamma_vasp_cmd)

        # construct jobs
        if job_type == "normal":
            jobs = [
                VaspJob(vasp_cmd, auto_npar=auto_npar, gamma_vasp_cmd=gamma_vasp_cmd)
            ]
        elif job_type == "normal_no_backup":
            jobs = [
                VaspJob(
                    vasp_cmd,
                    auto_npar=auto_npar,
                    gamma_vasp_cmd=gamma_vasp_cmd,
                    backup=False,
                )
            ]
        elif job_type == "double_relaxation_run":
            jobs = VaspJob.double_relaxation_run(
                vasp_cmd,
                auto_npar=auto_npar,
                ediffg=self.get("ediffg"),
                half_kpts_first_relax=self.get(
                    "half_kpts_first_relax", HALF_KPOINTS_FIRST_RELAX
                ),
            )
        elif job_type == "metagga_opt_run":
            jobs = VaspJob.metagga_opt_run(
                vasp_cmd,
                auto_npar=auto_npar,
                ediffg=self.get("ediffg"),
                half_kpts_first_relax=self.get(
                    "half_kpts_first_relax", HALF_KPOINTS_FIRST_RELAX
                ),
            )

        elif job_type == "full_opt_run":
            jobs = VaspJob.full_opt_run(
                vasp_cmd,
                auto_npar=auto_npar,
                ediffg=self.get("ediffg"),
                max_steps=9,
                half_kpts_first_relax=self.get(
                    "half_kpts_first_relax", HALF_KPOINTS_FIRST_RELAX
                ),
            )
        elif job_type == "neb":
            # TODO: @shyuep @HanmeiTang This means that NEB can only be run (i) in reservation mode
            # and (ii) when the queueadapter parameter is overridden and (iii) the queue adapter
            # has a convention for nnodes (with that name). Can't the number of nodes be made a
            # parameter that the user sets differently? e.g., fw_spec["neb_nnodes"] must be set
            # when setting job_type=NEB? Then someone can use this feature in non-reservation
            # mode and without this complication. -computron
            nnodes = int(fw_spec["_queueadapter"]["nnodes"])

            # TODO: @shyuep @HanmeiTang - I am not sure what the code below is doing. It looks like
            # it is trying to override the number of processors. But I tried running the code
            # below after setting "vasp_cmd = 'mpirun -n 16 vasp'" and the code fails.
            # (i) Is this expecting an array vasp_cmd rather than String? If so, that's opposite to
            # the rest of this task's convention and documentation
            # (ii) can we get rid of this hacking in the first place? e.g., allowing the user to
            # separately set the NEB_VASP_CMD as an env_variable and not rewriting the command
            # inside this.
            # -computron

            # Index the tag "-n" or "-np"
            index = [i for i, s in enumerate(vasp_cmd) if "-n" in s]
            ppn = int(vasp_cmd[index[0] + 1])
            vasp_cmd[index[0] + 1] = str(nnodes * ppn)

            # Do the same for gamma_vasp_cmd
            if gamma_vasp_cmd:
                index = [i for i, s in enumerate(gamma_vasp_cmd) if "-n" in s]
                ppn = int(gamma_vasp_cmd[index[0] + 1])
                gamma_vasp_cmd[index[0] + 1] = str(nnodes * ppn)

            jobs = [
                VaspNEBJob(
                    vasp_cmd,
                    final=False,
                    auto_npar=auto_npar,
                    gamma_vasp_cmd=gamma_vasp_cmd,
                )
            ]
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

        # construct handlers

        handler_group = self.get("handler_group", "default")
        if isinstance(handler_group, str):
            handlers = handler_groups[handler_group]
        else:
            handlers = handler_group

        if self.get("wall_time"):
            handlers.append(WalltimeHandler(wall_time=self["wall_time"]))

        if job_type == "neb":
            # CINEB vasprun.xml sometimes incomplete, file structure different
            validators = []
        else:
            validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(
            handlers,
            jobs,
            validators=validators,
            max_errors=max_errors,
            scratch_dir=scratch_dir,
            gzipped_output=gzip_output,
        )

        c.run()

        if os.path.exists(zpath("custodian.json")):
            if os.path.exists(zpath("FW_offline.json")):
                import json

                with open(zpath("custodian.json")) as f:
                    stored_custodian_data = {"custodian": json.load(f)}
            else:
                stored_custodian_data = {"custodian": loadfn(zpath("custodian.json"))}
            return FWAction(stored_data=stored_custodian_data)


@explicit_serialize
class RunBoltztrap(FiretaskBase):
    """
    Run Boltztrap directly. Requires vasprun.xml and OUTCAR to be in current dir.

    Required params:
        (none)

    Optional params:
        scissor: (float) scissor band gap amount in eV (i.e. new gap == scissor)
        tmax: (float) max temperature to evaluate (default = 1300K)
        tgrid: (float) temperature interval (default = 50K)
        doping: ([float]) doping levels you want to compute
        soc: (bool) whether the band structure is calculated with spin-orbit coupling or not
    """

    optional_params = ["scissor", "tmax", "tgrid", "doping", "soc"]

    def run_task(self, fw_spec):
        scissor = self.get("scissor", 0.0)
        tmax = self.get("tmax", 1300)
        tgrid = self.get("tgrid", 50)
        doping = self.get("doping", None)
        soc = self.get("soc", False)

        vasprun, outcar = get_vasprun_outcar(".", parse_dos=True, parse_eigen=True)
        bs = vasprun.get_band_structure()
        nelect = outcar.nelect
        runner = BoltztrapRunner(
            bs, nelect, scissor=scissor, doping=doping, tmax=tmax, tgrid=tgrid, soc=soc
        )

        runner.run(path_dir=os.getcwd())


@explicit_serialize
class RunNoVasp(FiretaskBase):
    """
    Do NOT run vasp. Do nothing.
    """

    def run_task(self, fw_spec):
        pass


@explicit_serialize
class RunVaspFake(FiretaskBase):
    """
    Vasp Emulator

    Required params:
        ref_dir (string): Path to reference vasp run directory with input files in the folder
           named 'inputs' and output files in the folder named 'outputs'.

    Optional params:
        params_to_check (list): optional list of incar parameters to check.
        check_incar (bool): whether to confirm the INCAR params (default: True)
        check_kpoints (bool): whether to confirm the KPOINTS params (default: True)
        check_poscar (bool): whether to confirm the POSCAR params (default: True)
        check_potcar (bool): whether to confirm the POTCAR params (default: True)
        clear_inputs (bool): whether to delete VASP input files after running (default: True)
    """

    required_params = ["ref_dir"]
    optional_params = [
        "params_to_check",
        "check_incar",
        "check_kpoints",
        "check_poscar",
        "check_potcar",
        "clear_inputs",
    ]

    def run_task(self, fw_spec):
        self._verify_inputs()
        if self.get("clear_inputs", True):
            self._clear_inputs()
        self._generate_outputs()

    def _verify_inputs(self):
        cwd = os.getcwd()
        user_incar = Incar.from_file(os.path.join(cwd, "INCAR"))
        input_path = os.path.join(self["ref_dir"], "inputs")

        # Carry out some BASIC tests.

        # Check INCAR
        if self.get("check_incar", True):
            ref_incar = Incar.from_file(
                os.path.join(self["ref_dir"], "inputs", "INCAR")
            )
            params_to_check = self.get("params_to_check", [])
            defaults = {"ISPIN": 1, "ISMEAR": 1, "SIGMA": 0.2}
            for p in params_to_check:
                if user_incar.get(p, defaults.get(p)) != ref_incar.get(
                    p, defaults.get(p)
                ):
                    raise ValueError(f"INCAR value of {p} is inconsistent!")

        # Check KPOINTS
        if self.get("check_kpoints", True):
            user_kpoints = Kpoints.from_file(os.path.join(cwd, "KPOINTS"))
            ref_kpoints = Kpoints.from_file(
                os.path.join(self["ref_dir"], "inputs", "KPOINTS")
            )
            if (
                user_kpoints.style != ref_kpoints.style
                or user_kpoints.num_kpts != ref_kpoints.num_kpts
            ):
                raise ValueError(
                    f"KPOINT files are inconsistent! Paths are:\n{cwd}\n{input_path} "
                    f"with kpoints {user_kpoints} and {ref_kpoints}"
                )

        # Check POSCAR
        if self.get("check_poscar", True):
            user_poscar = Poscar.from_file(os.path.join(cwd, "POSCAR"))
            ref_poscar = Poscar.from_file(
                os.path.join(self["ref_dir"], "inputs", "POSCAR")
            )
            if (
                user_poscar.natoms != ref_poscar.natoms
                or user_poscar.site_symbols != ref_poscar.site_symbols
            ):
                raise ValueError(
                    f"POSCAR files are inconsistent! Paths are:\n{cwd}\n{input_path}"
                )

        # Check POTCAR
        if self.get("check_potcar", True):
            user_potcar = Potcar.from_file(os.path.join(cwd, "POTCAR"))
            ref_potcar = Potcar.from_file(
                os.path.join(self["ref_dir"], "inputs", "POTCAR")
            )
            if user_potcar.symbols != ref_potcar.symbols:
                raise ValueError(
                    f"POTCAR files are inconsistent! Paths are:\n{cwd}\n{input_path}"
                )

        logger.info("RunVaspFake: verified inputs successfully")

    def _clear_inputs(self):
        for x in [
            "INCAR",
            "KPOINTS",
            "POSCAR",
            "POTCAR",
            "CHGCAR",
            "OUTCAR",
            "vasprun.xml",
        ]:
            p = os.path.join(os.getcwd(), x)
            if os.path.exists(p):
                os.remove(p)

    def _generate_outputs(self):
        # pretend to have run VASP by copying pre-generated outputs from reference dir to cur dir
        output_dir = os.path.join(self["ref_dir"], "outputs")
        for file_name in os.listdir(output_dir):
            full_file_name = os.path.join(output_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.getcwd())
        logger.info("RunVaspFake: ran fake VASP, generated outputs")

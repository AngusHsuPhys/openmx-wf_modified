from pymatgen.core.units import Ha_to_eV, bohr_to_ang

from atomate.openmx.database import openmxCalcDb

import os

import numpy as np
import pandas as pd

# chdir to the current file
os.chdir(os.path.dirname(os.path.realpath(__file__)))

db = openmxCalcDb.from_db_file("/workspaces/openmx-wf/Atomate/setting/db.json")



def check_force(taskid, plot=True):
    print(f"Checking forces for taskid: {taskid}")
    doc = db.collection.find_one({"task_id": taskid})

    calc_forces = doc["ase_calc"]["forces"]    
    calc_forces = np.array(calc_forces) * Ha_to_eV / bohr_to_ang

    if taskid != 1:
        prev_calc_force = doc["prev_calc"]["force"]
    else:
        prev_calc_force = doc["force"]
    prev_calc_force = np.array(prev_calc_force)
    print(f"prev_calc_force: {prev_calc_force}")
    print(f"calc_forces: {calc_forces}")
    print(f"shape of calc_forces: {calc_forces.shape}")
    print(f"shape of prev_calc_force: {prev_calc_force.shape}")
    # calculate the norm of the forces for each column in the forces array
    calc_forces = np.linalg.norm(calc_forces, axis=1)
    prev_calc_force = np.linalg.norm(prev_calc_force, axis=1)

    print(f"prev_calc_force: {prev_calc_force}")
    print(f"calc_forces: {calc_forces}")
    print(f"shape of calc_forces: {calc_forces.shape}")
    print(f"shape of prev_calc_force: {prev_calc_force.shape}")


    # get std, mean, max, min of the calculated forces and the previous calculated forces
    calc_forces_dict = {"max": np.max(calc_forces).round(4), "min": np.min(calc_forces).round(4), "mean": np.mean(calc_forces).round(4), "std": np.std(calc_forces).round(4)}
    prev_calc_forces_dict = {"max": np.max(prev_calc_force).round(4), "min": np.min(prev_calc_force).round(4), "mean": np.mean(prev_calc_force).round(4), "std": np.std(prev_calc_force).round(4)}

    # calculate the difference between the two forces
    diff = calc_forces - prev_calc_force

    # take abosulte value of the difference
    diff = np.abs(diff)

    #count the percentage of the difference that is greater or equal to 0.05 and less than 0.05
    num_greater_than_0_05 = np.sum(diff >= 0.05)
    num_less_than_0_05 = np.sum(diff < 0.05)
    print(f"len(diff): {len(diff)}")
    perc_greater_than_0_05 = (num_greater_than_0_05 / len(diff)) * 100
    perc_less_than_0_05 = (num_less_than_0_05 / len(diff)) * 100



    # get a heatmap of the difference between the two forces
    print(f"max diff (ev/ang): {np.max(diff).round(4)}")
    print(f"min diff (ev/ang): {np.min(diff).round(4)}")
    print(f"mean diff (ev/ang): {np.mean(diff).round(4)}")
    print(f"std diff (ev/ang): {np.std(diff).round(4)}")
    
    data = {"taskid": taskid, "max": np.max(diff).round(4), "min": np.min(diff).round(4), "mean": np.mean(diff).round(4), "std": np.std(diff).round(4),
            "perc_greater_than_0_05": perc_greater_than_0_05, "perc_less_than_0_05": perc_less_than_0_05, "Kgrid": doc["ase_calc"]["scf_kgrid"]}

    def get_heatmap(diff):
        # plot a heat map of the difference
        import matplotlib.pyplot as plt

        # get a new diff that only has two values, 0 and 1 where if the value is 0, it is less than 0.05, and if it is 1, it is greater than 0.05

        diff = np.where(diff <= 0.05, 0, 1)
        diff = diff.reshape(1, len(diff))
        plt.imshow(diff, cmap="viridis")
        plt.colorbar()
        plt.title(f"Force difference for taskid: {taskid}")
        plt.xlabel("atoms")
        plt.ylabel("forces")
        plt.savefig(f"force_diff_{taskid}.png")
        plt.show()



    if plot:
        get_heatmap(diff)

    return data, calc_forces_dict, prev_calc_forces_dict

def get_data():
    data = db.collection.find({}, {"task_id":1, "ase_calc.scf_kgrid":1})
    # give me a pandas dataframe of the data
    data = list(data)
    df = pd.DataFrame(data)

if __name__ == "__main__":
    diff_df = []
    calc_df = []
    prev_df = []

    for i in range(2, len(list(db.collection.find()))+1):
        diff, calc, prev = check_force(i, plot=False)
        diff_df.append(diff)

    # round the values in the dataframe to 2 decimal places
    print(pd.DataFrame(diff_df).round(2))

    # copy to clipboard
    pd.DataFrame(diff_df).round(2).to_clipboard()
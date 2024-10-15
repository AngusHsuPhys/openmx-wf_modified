from atomate.openmx.database import openmxCalcDb
from monty.serialization import dumpfn, loadfn
from prettytable import PrettyTable
import pandas as pd

db = openmxCalcDb.from_db_file("/workspaces/openmx-wf/docker-img/get-data/db.json", admin=True)


def find_duplicates():
    """
    Find the duplicates in the database based on the i["prev_vasp_calc"]["info"]["energy"]
    """
    collection = db.collection # replace with your collection name
    pipeline = [
        {
            "$group": {
                "_id": "$prev_vasp_calc.info.energy",  # group by energy
                "task_ids": { "$push": "$task_id" },  # collect task ids
                "count": { "$sum": 1 }  # count occurrences
            }
        },
        {
            "$match": {
                "count": { "$gt": 1 }  # only show duplicates
            }
        },
        # add a new field deeph_raw_len by checking the "calcs_reversed.deeph_raw" length of each task_id in the task_ids
        {
            "$lookup": {
                "from": collection.name,
                "localField": "task_ids",
                "foreignField": "task_id",
                "as": "matched_docs"
            }
        },
        {
            "$unwind": "$matched_docs"
        },
        {
            "$project": {
                "task_id": "$matched_docs.task_id",
                "deeph_raw": { "$arrayElemAt": [ "$matched_docs.calcs_reversed.deeph_raw", 0 ] },
            }
        },
        {
            "$project": {
                "task_id": 1,
                "deeph_raw": { "$objectToArray": "$deeph_raw" }
            }
        },
        {
            "$project": {
                "task_id": 1,
                "deeph_raw": { "$map": { "input": "$deeph_raw", "as": "item", "in": "$$item.k" } }
            }
        },
        {
            "$project": {
                "task_id": 1,
                "deeph_raw_len": {
                    "$cond": {
                        "if": { "$isArray": "$deeph_raw" },
                        "then": { "$size": "$deeph_raw" },
                        "else": None
                    }
                }
            }
        },
        # group by _id and collect the task_ids and deeph_raw_len
        {   
            "$group": {
                "_id": "$_id",
                "task_ids": { "$push": "$task_id" },
                "deeph_raw_len": { "$push": "$deeph_raw_len" }
            }
        },

    ]

    duplicates = list(collection.aggregate(pipeline))

    # make this list a table with prettytable and don't specify the field names
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = duplicates[0].keys()
    for i in range(len(duplicates)):
        x.add_row(duplicates[i].values())
    print(x)
    return duplicates

# define a function that reads the duplicates and remove the duplicates by the task_id and deeph_raw_len being None
def remove_duplicates(duplicates):
    for duplicate in duplicates:
        task_ids = duplicate["task_ids"]
        deeph_raw_len = duplicate["deeph_raw_len"]
        # remove the task_id with the deeph_raw_len being None
        for i in range(len(deeph_raw_len)):
            if deeph_raw_len[i] is None:
                print(f"Removing duplicate task_id due to deeph_raw_len being None: {task_ids[i]}")
                db.collection.delete_one({"task_id": task_ids[i]})
            # remove those with the deeph_raw_len being not 14
            elif deeph_raw_len[i] != 14:
                print(f"Removing duplicate task_id due to deeph_raw_len being not 14: {task_ids[i]}")
                db.collection.delete_one({"task_id": task_ids[i]})
            # finally keep the first task_id by deleting the rest
            else:
                print(f"Keeping the first task_id: {task_ids[i]}")
                db.collection.delete_one({"task_id": task_ids[i]})

def get_grouped_table():
    pipeline = [
    {
        "$group": {
            "_id": "$prev_vasp_calc.info.energy",  # group by energy
            "task_ids": { "$push": "$task_id" },  # collect task ids
            "count": { "$sum": 1 }  # count occurrences
        }
    },
    ]
    grouped_table = list(db.collection.aggregate(pipeline))
    df = pd.DataFrame(grouped_table)
    # print out any rows with count > 1
    print(df[df["count"] == 1])



def get_energy_table_from_db():
    """
    Get the energy table from the database and the field "prev_vasp_calc.info.energy"
    """
    collection = db.collection
    pipeline = [
        {
            "$project": {
                "task_id": 1,
                "prev_vasp_calc.info.energy": 1
            }
        }
    ]
    energy_table = list(collection.aggregate(pipeline))

    # make a df whose columns are task_id and energy
    energy_df = []
    for i in range(len(energy_table)):
        task_id = energy_table[i]["task_id"]
        energy = energy_table[i]["prev_vasp_calc"]["info"]["energy"]
        energy_df.append([task_id, energy])
    df = pd.DataFrame(energy_df, columns=["task_id", "energy"])
    print(df)
    return df

def find_missing_calcs():
    # first read the json /workspaces/openmx-wf/SrGeSi/initial-train.json
    target = loadfn("/workspaces/openmx-wf/SrGeSi/initial-train.json")
    
    energy_df = []
    for key in target.keys():
        energy = target[key]["info"]["energy"]
        energy_df.append(energy)
    # create a dataframe from the energy_df, and key as another column uid
    json_df = pd.DataFrame(energy_df, columns=["energy"], index=target.keys())
    json_df["uid"] = json_df.index

    # get the energy table from the database
    energy_df_from_db = get_energy_table_from_db().round(8)

    # compare the two dataframes and find the missing calculations in the database
    # when comparing the two dataframes, make sure the rounding is the same
    missing_calcs_df = json_df[~json_df["energy"].isin(energy_df_from_db["energy"])]

    calculated_df = json_df[json_df["energy"].isin(energy_df_from_db["energy"])]
    
    # add taskid to the calculated_df from comparing the energy
    task_ids = []
    for energy in calculated_df["energy"]:
        task_id = energy_df_from_db[energy_df_from_db["energy"] == energy]["task_id"].values[0]
        task_ids.append(task_id)
    calculated_df["task_id"] = task_ids

    return missing_calcs_df, calculated_df

def update_db_with_uid(calculated_df):
    collection = db.collection
    for i in range(len(calculated_df)):
        task_id = calculated_df.iloc[i]["task_id"]
        uid = calculated_df.iloc[i]["uid"]
        print(f"Updating the database with the task_id: {task_id} and uid: {uid}")
        # make sure task_id is and uid is int
        task_id = int(task_id)
        uid = int(uid)
        collection.update_one({"task_id": task_id}, {"$set": {"uid": uid}})
    print("Updated the database with the uid")


def gen_calculation_dict(target, missing_calcs_df):
    """
    Generate the calculation dict by keeping the keys in missing_calcs_df in target
    """
    calculation_dict = {}
    for key in missing_calcs_df.index:
        calculation_dict[key] = target[key]
    return calculation_dict


class main:
    @classmethod
    def gen_missing_calc_json(cls):
        missing_calcs_df, _ = find_missing_calcs()
        print(missing_calcs_df.count())

        # generate the calculation dict
        calculation_dict = gen_calculation_dict(loadfn("/workspaces/openmx-wf/SrGeSi/initial-train.json"), missing_calcs_df)
        # print len of calculation_dict
        dumpfn(calculation_dict, "/workspaces/openmx-wf/SrGeSi/missing_calcs.json")

    @classmethod
    def update_db_with_uid(cls):
        _, calculated_df = find_missing_calcs()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            executor.map(update_db_with_uid, [calculated_df])


if __name__ == "__main__":
    # duplicates = find_duplicates()
    # remove_duplicates(duplicates)
    # get_energy_table_from_db()
    missing_calcs_df, calculated_df = find_missing_calcs()
    print(calculated_df.count())
    # get_grouped_table()
    
    # main.gen_missing_calc_json()
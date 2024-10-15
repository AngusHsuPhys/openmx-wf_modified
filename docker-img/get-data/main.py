#!/usr/bin/env python3

from atomate.openmx.database import openmxCalcDb
from monty.serialization import dumpfn
import argparse
import concurrent.futures
import os
import threading

def get_files(task_id, store_in, db="db.json", path="openmx_raw"):
    db = openmxCalcDb.from_db_file(db, admin=True)
    entry = db.collection.find_one({"task_id": task_id})


        
    # write raw
    keys = entry["calcs_reversed"][0][path].keys()
    keys = [key.split("_fs_id")[0] for key in keys if "_fs_id" in key]
    print(f"Keys: {keys}")
    store_in = f"{store_in}/{path}/{task_id}"

    def write_output(key):
        out = db.get_openmx_output(task_id, key, path)
        # make the directory if it does not exist
        print(f"---> {store_in} <---")
        if not os.path.exists(f"{store_in}"):
            os.makedirs(f"{store_in}")
            print(f"Directory {store_in} created")

            # write the deeph file to the store_in directory f"{store_in}/{task_id}/{path}/info.json"
            if path == "deeph_raw":
                print(f"---> {store_in}/info.json <---")
                deep_info = entry["deeph"]
                print(deep_info)
                dumpfn(deep_info, f"{store_in}/info.json", indent=4)

        if path == "openmx_rst":
            key = key.replace("_", ".", 1)
        else:
        # replace the last "_" with a "." in key
            key = key[::-1].replace("_", ".", 1)[::-1]

        with open(f"{store_in}/{key}", "wb") as f:
            f.write(out)

    # Create a lock
    lock = threading.Lock()
    def write_output_with_lock(key):
        # Acquire the lock
        with lock:
            # Call the original function
            write_output(key)

    # Use a ThreadPoolExecutor to run the loop in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(write_output_with_lock, keys)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Get the openmx and deephe3 files from the database")

    # Add the arguments
    parser.add_argument('-t', '--taskid', type=int, help='The task id of the calculation', required=True)
    parser.add_argument('-s', '--store_in', type=str, help='The directory to store the files in', default="/data")
    parser.add_argument('-d', '--db_file', type=str, help='The path to the database file', default="/deeph/db.json")
    parser.add_argument('-p', '--path', type=str, help='3 options: deeph_raw, openmx_rst, openmx_raw', required=True)

    # Parse the arguments
    args = parser.parse_args()

    get_files(args.taskid, args.store_in, args.db_file, args.path)


if __name__ == "__main__":
    main()  
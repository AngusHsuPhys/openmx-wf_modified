import os
import sys
from monty.serialization import loadfn, dumpfn

def create_fworker(port, project_name, instance_name):
    # Define the base path for the project
    base_path = '/home/j.tsai/config/project/{}'.format(project_name)
    
    # Ensure port is an integer
    port = int(port)
    
    # Create the directory for the instance, if it doesn't exist
    instance_path = os.path.join(base_path, instance_name)
    os.makedirs(instance_path, exist_ok=True)
    
    # Load the configuration files
    db_config = loadfn("db.json")
    fworker_config = loadfn("my_fworker.yaml")
    qadapter_config = loadfn("my_qadapter.yaml")
    launchpad_config = loadfn("my_launchpad.yaml")
    
    # Update the configurations with the project and instance specific details
    launchpad_config.update({"name": project_name, "port": port})
    db_config.update({"database": project_name, "collection": instance_name, "port": port})
    fworker_config.update({"category": instance_name})
    fworker_config["env"].update({"db_file": os.path.join(instance_path, "db.json")})
    qadapter_config.update({"rocket_launch": "rlaunch -c {} singleshot".format(instance_path)})
    
    # Save the updated configurations back to their respective files
    dumpfn(db_config, os.path.join(instance_path, "db.json"), indent=4)
    dumpfn(fworker_config, os.path.join(instance_path, "my_fworker.yaml"))
    dumpfn(qadapter_config, os.path.join(instance_path, "my_qadapter.yaml"))
    dumpfn(launchpad_config, os.path.join(instance_path, "my_launchpad.yaml"))
    
    # Create a directory in the scratch space for the project and instance
    os.makedirs(os.path.join("/home/j.tsai/scratch", project_name, instance_name), exist_ok=True)

# Call the function with command line arguments
create_fworker(sys.argv[-3], sys.argv[-2], sys.argv[-1])
1. One can use `create_fworker.py` to create customized configuration files of a project. It takes `my_fworker.yaml`, `my_launchpad.yaml`, `my_qadapter.yaml`, and `db.json` as the template files.

2. Main replacement fields in the template files are:
    ```python
    # Update the configurations with the project and instance specific details
    launchpad_config.update({"name": project_name, "port": port})
    db_config.update({"database": project_name, "collection": instance_name, "port": port})
    fworker_config.update({"category": instance_name})
    fworker_config["env"].update({"db_file": os.path.join(instance_path, "db.json")})
    qadapter_config.update({"rocket_launch": "rlaunch -c {} singleshot".format(instance_path)})
    ```
    where, 
    - `project_name` is the name of the project.
    - `instance_name` is the name of calculation related to the project.
    - `port` is the port number for the MongoDB server.


3. How to run: `python create_fworker.py <port> <project_name> <instance_name>`  

4. Run this on the compute sites.
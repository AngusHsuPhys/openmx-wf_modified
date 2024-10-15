1. To establish a connection to the Sinica database, use the following SSH configuration:
   ```shell
   Host db-sinica 
      HostName login.discovery.neu.edu
      User j.tsai
      LocalForward 12345 hercules.phys.sinica.edu.tw:12345


2. Configure `my_launchpad.yaml` and create a `logs` directory on your local machine. This setup is necessary for launching and monitoring workflows.
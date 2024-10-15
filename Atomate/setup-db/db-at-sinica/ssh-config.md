# To connect to the db server at sinica, you need to have the following in your ~/.ssh/config file:

```
Host JumpHost
  HostName JumpHost
  User SomeUser

Host sinica
    Hostname hercules.phys.sinica.edu.tw
    ProxyJump JumpHost
    User tsaie79

Host cont-sinica
    Hostname trgn01
    User tsaie79
    Proxyjump sinica

Host db-sinica
    HostName trgn01
    User tsaie79
    ProxyJump cont-sinica
    LocalForward 27018 localhost:27017

```
* This will allow you to connect to the db server at sinica by typing `ssh -Nf db-sinica` in your terminal.
* Make sure that user can login to `JumpHost`, `sinica`, and `cont-sinica` without password.


# Description of files in this directory
1. `ssh-config.md`: This file contains the description of the ssh config file.
2. `mongod.sh`: This file contains the command to start the mongod server.
3. `mongosh.sh`: This file contains the command to start the mongosh client.
4. `op-db.sh`: This file contains the commands to create and delete the database and its users.
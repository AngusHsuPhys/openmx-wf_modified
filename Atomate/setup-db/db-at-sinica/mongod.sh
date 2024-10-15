#!/bin/bash

# creaete mongodb.log in $HOME/mongodb/log/ if not exist
mkdir -p $DB_PATH/log/
touch $DB_PATH/log/mongodb.log

export DB_PATH="/home/tsaie79/db/MongoDB" 
export MONGODB_IMAGE=$HOME/mongo_latest.sif

singularity exec -B $DB_PATH/data/db:/data/db -B $DB_PATH/log/:/var/log/ $MONGODB_IMAGE mongod --logpath /var/log/mongodb.log --dbpath /data/db --bind_ip_all &

#!/bin/bash

export DB_PATH="/shared/Project/NCKU/Deep-SK/data/MongoDB"
export MONGODB_IMAGE=$HOME/mongo_latest.sif

export TARGET_PATH="/shared/Project/NCKU/Deep-SK/datadb/MongoDB"


singularity exec -B $DB_PATH/data/db:/data/db -B $DB_PATH/log/:/var/log/ -B $TARGET_PATH:/out $MONGODB_IMAGE mongodump --host localhost --port 27017 --out /out
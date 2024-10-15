#!/bin/bash

export MONGODB_IMAGE=$HOME/mongo_latest.sif

export TARGET_PATH="/shared/Project/NCKU/Deep-SK/datadb/MongoDB"


singularity exec -B $TARGET_PATH/data/db:/data/db -B $TARGET_PATH/log/:/var/log/ -B $TARGET_PATH/mig:/out $MONGODB_IMAGE mongorestore --host localhost --port 27017 /out
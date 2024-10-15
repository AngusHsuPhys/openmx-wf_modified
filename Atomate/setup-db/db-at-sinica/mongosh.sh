#!/bin/bash

export MONGODB_IMAGE=$HOME/mongo_latest.sif

singularity exec $MONGODB_IMAGE mongosh
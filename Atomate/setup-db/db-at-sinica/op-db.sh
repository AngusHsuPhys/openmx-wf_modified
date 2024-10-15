#!/bin/bash

# Check if -h flag is provided
if [ "$1" = "-h" ]
then
    echo "Usage: ./create-db.sh -c|-d project-name"
    echo "-c: create database and users"
    echo "-d: delete database and users"
    echo "project-name: name of the project"
    exit 0
fi

# Check if operation is provided
if [ -z "$1" ]
then
    echo "Please provide an operation (-c or -d)."
    exit 1
fi

# Check if project name is provided
if [ -z "$2" ]
then
    echo "Please provide a project name."
    exit 1
fi

# Set the MongoDB image path
export MONGODB_IMAGE=$HOME/mongo_latest.sif

# Set the operation from the first command line argument
export operation="$1"

# Set the database name from the second command line argument
export name_project="$2"

# Perform the requested operation
if [ "$operation" = "-c" ]
then
    # Create the users
    singularity exec $MONGODB_IMAGE mongosh --eval "db.getSiblingDB('$name_project').createUser({user: 'Hsin', pwd: 'Hsin', roles: [{role: 'readWrite', db: '$name_project'}]});"
    singularity exec $MONGODB_IMAGE mongosh --eval "db.getSiblingDB('$name_project').createUser({user: 'Hsin_ro', pwd: 'Hsin', roles: [{role: 'read', db: '$name_project'}]});"
elif [ "$operation" = "-d" ]
then
    # Delete the database
    singularity exec $MONGODB_IMAGE mongosh --eval "db.getSiblingDB('$name_project').dropDatabase();"
    # Delete the users in the database
    singularity exec $MONGODB_IMAGE mongosh --eval "db.getSiblingDB('$name_project').dropUser('Hsin'); db.getSiblingDB('$name_project').dropUser('Hsin_ro');"
else
    echo "Invalid operation. Please provide '-c' or '-d'."
    exit 1
fi
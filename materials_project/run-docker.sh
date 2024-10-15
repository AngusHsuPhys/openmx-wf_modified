#!/bin/bash

docker run -it --rm -v $(pwd):/mp --net=host mp-api
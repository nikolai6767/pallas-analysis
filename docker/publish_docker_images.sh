#!/bin/bash

docker login registry.gitlab.inria.fr
docker build -t registry.gitlab.inria.fr/pallas/pallas -f Dockerfile_pallas_test .
docker push registry.gitlab.inria.fr/pallas/pallas

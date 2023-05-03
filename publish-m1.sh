#!/bin/sh
VERSION=`python setup.py get_project_version | tail -n 1`
docker build -t pikalab/demo-lime:latest-apple-m1 .
docker tag pikalab/demo-lime:latest-apple-m1 pikalab/demo-lime:$VERSION-apple-m1
docker push pikalab/demo-lime:latest-apple-m1
docker push pikalab/demo-lime:$VERSION-apple-m1

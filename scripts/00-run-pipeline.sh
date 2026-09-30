#!/usr/bin/bash
./scripts/pipeline/setup.sh
./scripts/pipeline/slicing.sh
./scripts/pipeline/training-ce.sh
./scripts/pipeline/stitching.sh


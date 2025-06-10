#!/bin/bash

# Run the dynamics_toy_example module with Python -m
rm -rf images
python3 -m mf_estimation.finite_population_grid_nav.test_grid_nav
mkdir images
mv *png images
python3 -m mf_estimation.gif
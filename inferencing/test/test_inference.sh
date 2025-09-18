#!/usr/bin/env bash
# Ensure random values work for inference
python3 inferencing/test/test_inference.py

# Ensure pre-engineered features path works
python3 inferencing/test/test_pre-engineered_features.py -n 100

#!/bin/bash

for file in INPUT/*; do
    python3 segment2.py "$file"
done

for file in INPUT2/*; do
    python3 segment2.py "$file"
done

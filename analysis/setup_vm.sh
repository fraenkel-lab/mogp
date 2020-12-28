#!/bin/bash

sudo apt update
sudo apt install python3-pip
sudo apt install python3-venv
python3 -m venv mogp/mogpvenv
source mogp/mogpvenv/bin/activate
pip install -r mogp/analysis/analysis_requirements.txt
python mogp/setup.py install

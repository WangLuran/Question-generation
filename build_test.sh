#!/bin/bash
source ~/.bashrc
conda activate Jtest
python ChatGPT_QG.py --api_key=""--save_path="chatgpt-qg.json"

conda deactivate

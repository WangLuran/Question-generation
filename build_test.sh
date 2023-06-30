#!/bin/bash
source ~/.bashrc
conda activate Jtest
python ChatGPT_QG.py --api_key="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" --save_path="chatgpt-qg.json"

conda deactivate

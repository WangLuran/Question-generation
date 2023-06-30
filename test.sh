source ~/.bashrc
conda activate Jtest

python GPT2_QG_test.py --"model_path"="gpt2_gen_seed1.pt" --"predictions_save_path"="./"

conda deactivate

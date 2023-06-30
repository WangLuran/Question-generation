source ~/.bashrc
conda activate Jtest

python GPT2/test.py --"model_path"="gpt2_gen_seed1.pt" --"predictions_save_path"="./"

conda deavtivate

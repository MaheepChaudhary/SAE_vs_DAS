echo "Running eval for Bloom SAE"
python3.11 eval_sae_main.py -d "cuda:1" -met "sae masking openai" -bs 16  

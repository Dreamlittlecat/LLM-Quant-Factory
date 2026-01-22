# python -u evaluate_flan.py -m ../gptq/weights/bloom-560m > flan_bloom.log
ngpu=1
data_dir=/home/skl/lin/codes/qlora/eval/mmlu-raw/data
save_dir=tmp/
ntrain=5
# python -u evaluate.py --data_dir ${data_dir} --save_dir ${save_dir} --ntrain ${ntrain} > run.log
# CUDA_VISIBLE_DEVICES=7, python -u evaluate_flan.py -m huggyllama/llama-7b --data_dir ${data_dir} --save_dir ${save_dir} --ntrain ${ntrain} > run.log
CUDA_VISIBLE_DEVICES=7, python -u evaluate_hf.py -m huggyllama/llama-13b > run.log
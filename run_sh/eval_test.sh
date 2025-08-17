if [ ! -d "xjh_logs" ]; then
    mkdir xjh_logs
fi
# export MODEL_NAME=opt-125m && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1
# export MODEL_NAME=opt-350m && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1
# export MODEL_NAME=opt-1.3b && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1
# export MODEL_NAME=opt-2.7b && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1
# export MODEL_NAME=opt-6.7b && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1
# export MODEL_NAME=opt-13b && python eval_test.py --model /data/xjh/model_weight/opt//${MODEL_NAME} >>xjh_logs/${MODEL_NAME}.log 2>&1

#export MODEL_NAME=llama3.2-1b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1
export MODEL_NAME=llama3.2-3b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1
export MODEL_NAME=llama3-8b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1
export MODEL_NAME=llama2-7b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1
export MODEL_NAME=llama2-13b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1
export MODEL_NAME=llama-13b && python eval_test.py --model /data/xjh/model_weight/llama//${MODEL_NAME} >>xjh_logs/no_quant/${MODEL_NAME}.log 2>&1



export MODEL_NAME=llama3.2-1b && python eval_test.py --model /home/xjh/research/LLM-quant/ParetoQ/tmp/llama/models/1B-finetuned
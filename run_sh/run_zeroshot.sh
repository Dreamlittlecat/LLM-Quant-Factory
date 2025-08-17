
# MODELS=(
#     "opt-125m"
#     "opt-350m"
#     "opt-1.3b"
#     "opt-2.7b"
#     "opt-6.7b"
#     "opt-13b"
# )
# MODEL_DIR="/data/xjh/model_weight/opt"


DEVICE="cuda:0"
TASKS="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,openbookqa"
LOG_DIR="./xjh_logs/zeroshot_noquant"

# # 创建日志目录（如果不存在）
# mkdir -p "$LOG_DIR"

# # 遍历配置

# for MODEL in "${MODELS[@]}"; do
#     LOG_FILE="$LOG_DIR/${MODEL}"
#     echo "Running model $MODEL ..."
#     python zero_shot.py --model "$MODEL_DIR/$MODEL" --tasks "$TASKS"  --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1   
# done

TASKS="piqa,openbookqa"
MODELS=(
    # "llama3.2-1b"
    # "llama3.2-3b"
    # "llama3-8b"
    # "llama2-7b"
    # "llama2-13b"
    "llama-7b"

)
MODEL_DIR="/data/xjh/model_weight/llama"

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="$LOG_DIR/${MODEL}"
    echo "Running model $MODEL ..."
    python zero_shot.py --model "$MODEL_DIR/$MODEL" --tasks "$TASKS"  --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1   
done

# LOG_DIR="./xjh_logs/zeroshot_sdm"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/llama3-8b"
# echo "Running model $MODEL ..."
# #piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,
# python zero_shot.py --model "/home/xjh/research/AI_xjh_research/model_fakequant_eval/save/sdm/llama3-8b_c4_osr2.0_first2False_hadamardTrue_blocksize128" --tasks "openbookqa"  --device "cuda:0" >> "${LOG_FILE}.log" 2>&1   
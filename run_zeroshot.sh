

DEVICE="cuda:0"
TASKS="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,openbookqa"
LOG_DIR="./examples_logs/zeroshot_noquant"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# # 遍历配置


TASKS="piqa,openbookqa"
#模型路径
MODEL_DIR="/data/xjh/model_weight/llama"
MODELS=(
     "llama3.2-1b"
    # "llama3.2-3b"
    # "llama3-8b"
    # "llama2-7b"
    # "llama2-13b"
    #"llama-7b"
)


for MODEL in "${MODELS[@]}"; do
    LOG_FILE="$LOG_DIR/${MODEL}"
    echo "Running model $MODEL ..."
    python zero_shot_eval.py --model "$MODEL_DIR/$MODEL" --log_dir "$LOG_DIR" --tasks "$TASKS"  --device "$DEVICE" 
done

#METHOD=complex1
METHOD=complex1

#METHOD=complex2_adakmeans
#配置参数
# MODELS=(
#     "opt-125m"
#     "opt-350m"
#     "opt-1.3b"
#     "opt-2.7b"
#     "opt-6.7b"
#     "opt-13b"
# )
# MODEL_DIR="/data/xjh/model_weight/opt"
# MODELS=(
#     "llama3.2-1b"
#     "llama3.2-3b"
#     "llama3-8b"
#     "llama2-7b"
#     "llama2-13b"

# )
# MODEL_DIR="/data/xjh/model_weight/llama"



MODELS=(
    # "/data/xjh/model_weight/llama/llama3.2-1b"
    # "/data/xjh/model_weight/llama/llama3.2-3b"
    # "/data/xjh/model_weight/llama/llama3-8b"
    #  "/data/xjh/model_weight/llama/llama2-7b"
    # "/data/xjh/model_weight/llama/llama2-13b"
     "/data/xjh/model_weight/llama/llama-7b"
     "/data/xjh/model_weight/llama/llama-13b"
    # "/data/xjh/model_weight/opt/opt-125m"
    # "/data/xjh/model_weight/opt/opt-350m"
    # "/data/xjh/model_weight/opt/opt-1.3b"
    # "/data/xjh/model_weight/opt/opt-2.7b"
    #  "/data/xjh/model_weight/opt/opt-6.7b"
    #  "/data/xjh/model_weight/opt/opt-13b"
)

#BLOCKSIZES=(128 )
BLOCKSIZE=128
DEVICE="cuda:1"

DATASET="c4"
LOG_DIR="./xjh_logs/${METHOD}_${DATASET}"
#LOG_DIR="./xjh_logs/${METHOD}_test_no_svd_${DATASET}"
CLUSTER_M=4
CLUSTER_P=8
GROUPSIZE=-4
GROUPSIZE=-1
GROUPSIZES=(-1 -2 -4)
# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 遍历配置
# for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
for GROUPSIZE in "${GROUPSIZES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        model_name=$(basename "$MODEL")
        # 配置日志文件名
        #LOG_FILE="$LOG_DIR/${MODEL}-complex2-quant_cluster_m_${CLUSTER_M}_cluster_p_${CLUSTER_P}"
        LOG_FILE="$LOG_DIR/${model_name}-${METHOD}-quant_cluster_m_${CLUSTER_M}_cluster_p_${CLUSTER_P}"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi
        LOG_FILE+="_groupsize_${GROUPSIZE}"
  
        echo "Running ${METHOD} quant for model $model_name and blocksize $BLOCKSIZE..."
        python3 run_complex.py "$MODEL" "$DATASET" --blocksize "$BLOCKSIZE"  --cluster_m $CLUSTER_M --cluster_p $CLUSTER_P --groupsize $GROUPSIZE --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1
    done
done


# python3 run_complex.py /data/xjh/model_weight/opt/opt-125m c4 --blocksize 128 --cluster_m 2 --cluster_p 8 --device "cuda:1" 
# python3 run_complex.py /data/xjh/model_weight/opt/opt-1.3b c4 --blocksize 128 --cluster_m 2 --cluster_p 8 --device "cuda:1" 
# python3 run_complex.py /data/xjh/model_weight/opt/opt-2.7b c4  --blocksize 128 --cluster_m 2 --cluster_p 8 --device "cuda:1"
# python3 run_complex.py /data/xjh/model_weight/opt/opt-6.7b c4  --blocksize 128 --cluster_m 2 --cluster_p 8 --device "cuda:1" 
# python3 run_complex.py /data/xjh/model_weight/opt/opt-13b c4  --blocksize 128 --cluster_m 2 --cluster_p 8 --device "cuda:1" 

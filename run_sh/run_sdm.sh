# version: 1.0
# # 配置参数
# # MODELS=(
# #     "opt-125m"
# #     "opt-350m"
# #     "opt-1.3b"
# #     "opt-2.7b"
# #      "opt-6.7b"
# #     "opt-13b"
# # )
# # MODEL_DIR="/data/xjh/model_weight/opt"

# MODELS=(
#     # "llama3.2-1b"
#     #  "llama3.2-3b"
#     #  "llama3-8b"
#      "llama-7b"
#     # "llama-13b"
#     # "llama2-7b"
#     # "llama2-13b"

# )
# MODEL_DIR="/data/xjh/model_weight/llama"
# BLOCKSIZES=(128 )
# DEVICE="cuda:1"

# DATASET="c4"
# HADAMARD="--hadamard"
# OSR=2

# #LOG_DIR="./xjh_logs/sdm_multilayer_test${DATASET}"
# LOG_DIR="./xjh_logs/sdm_config${DATASET}"
# # 创建日志目录（如果不存在）
# mkdir -p "$LOG_DIR"

# # 遍历配置
# for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
#     for MODEL in "${MODELS[@]}"; do
#         # 配置日志文件名
#         LOG_FILE="$LOG_DIR/${MODEL}-sdm-quant_hadamard-osr_${OSR}"
#         if [ "$BLOCKSIZE" -eq -1 ]; then
#             LOG_FILE+="_noblock"
#         else
#             LOG_FILE+="_blocksize_${BLOCKSIZE}"
#         fi

#         # 启用 GPTQ
#         echo "Running SDM with GPTQ for model $MODEL and blocksize $BLOCKSIZE OSR $OSR..."
#         python3 run_sdm.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --osr $OSR $HADAMARD --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1
        
#         # #first2
#         #LOG_FILE="$LOG_DIR/${MODEL}-sdm-quant_hadamard-osr_${OSR}_first2"
#         # echo "Running SDM with GPTQ for model $MODEL and blocksize $BLOCKSIZE OSR $OSR first2..."
#         # python3 run_sdm.py "$MODEL_DIR/$MODEL" "$DATASET" --blocksize "$BLOCKSIZE" --osr $OSR $HADAMARD --first2  --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

# done





#version: 2
#!/bin/bash
# filepath: /home/xjh/research/AI_xjh_research/model_fakequant_eval/run_sh/run_sdm_combined.sh

# Define array of model paths
models=(
    # "/data/xjh/model_weight/llama/llama3.2-1b"
    # "/data/xjh/model_weight/llama/llama3.2-3b"
     "/data/xjh/model_weight/llama/llama3-8b"
    #  "/data/xjh/model_weight/llama/llama2-7b"
    # "/data/xjh/model_weight/llama/llama2-13b"
    # "/data/xjh/model_weight/llama/llama-7b"
    # "/data/xjh/model_weight/llama/llama-13b"
    #"/data/xjh/model_weight/llama/vicuna-7b-1.1"
    # "/data/xjh/model_weight/opt/opt-125m"
    # "/data/xjh/model_weight/opt/opt-350m"
    # "/data/xjh/model_weight/opt/opt-1.3b"
    # "/data/xjh/model_weight/opt/opt-2.7b"
    #  "/data/xjh/model_weight/opt/opt-6.7b"
    #  "/data/xjh/model_weight/opt/opt-13b"
)

# Common parameters
BLOCKSIZES=(128)
DEVICE="cuda:0"
DATASET="c4"
HADAMARD="--hadamard"
OSR=2.00

# Log directory
LOG_DIR="./xjh_logs/sdm_test_v6_${DATASET}"
mkdir -p "$LOG_DIR"

# Function to run SDM for a model
run_sdm() {
    model_path=$1
    model_name=$(basename "$model_path")
    
    echo "----------------------------------------"
    echo "Processing model: $model_name"
    
    for BLOCKSIZE in "${BLOCKSIZES[@]}"; do
        # Configure log file name
        LOG_FILE="$LOG_DIR/${model_name}-sdm-quant_hadamard-osr_${OSR}"
        if [ "$BLOCKSIZE" -eq -1 ]; then
            LOG_FILE+="_noblock"
        else
            LOG_FILE+="_blocksize_${BLOCKSIZE}"
        fi

        # Run with standard settings
        echo "Running SDM with GPTQ for model $model_name, blocksize $BLOCKSIZE, OSR $OSR..."
        python3 run_sdm.py "$model_path" "$DATASET" \
            --blocksize "$BLOCKSIZE" \
            --osr $OSR \
            $HADAMARD \
            --config_path "./output/${model_name}/linear_osr_${OSR}.json" \
            --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        # LOG_FILE+="_no_config"
        # echo "Running SDM with GPTQ for model $model_name, blocksize $BLOCKSIZE, OSR $OSR... no config"
        # python3 run_sdm.py "$model_path" "$DATASET" \
        #     --blocksize "$BLOCKSIZE" \
        #     --osr $OSR \
        #     $HADAMARD \
        #     --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        # LOG_FILE="$LOG_DIR/${model_name}-sdm-quant-osr_${OSR}"
        # if [ "$BLOCKSIZE" -eq -1 ]; then
        #     LOG_FILE+="_noblock"
        # else
        #     LOG_FILE+="_blocksize_${BLOCKSIZE}"
        # fi

        # echo "Running SDM with GPTQ for model $model_name, blocksize $BLOCKSIZE, OSR $OSR..."
        # python3 run_sdm.py "$model_path" "$DATASET" \
        #     --blocksize "$BLOCKSIZE" \
        #     --osr $OSR \
        #     --config_path "./output/${model_name}/linear_osr_${OSR}.json" \
        #     --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1  
             
        # LOG_FILE+="_no_config"
        # echo "Running SDM with GPTQ for model $model_name, blocksize $BLOCKSIZE, OSR $OSR... no config"
        # python3 run_sdm.py "$model_path" "$DATASET" \
        #     --blocksize "$BLOCKSIZE" \
        #     --osr $OSR \
        #     --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1   

        # Uncomment to run with first2 option
        # LOG_FILE="$LOG_DIR/${model_name}-sdm-quant_hadamard-osr_${OSR}_first2"
        # echo "Running SDM with first2 for model $model_name, blocksize $BLOCKSIZE, OSR $OSR..."
        # python3 run_sdm.py "$model_path" "$DATASET" \
        #     --blocksize "$BLOCKSIZE" \
        #     --osr $OSR \
        #     $HADAMARD \
        #     --first2 \
        #     --config_path "./output/${model_name}/linear_osr_${OSR}.json" \
        #     --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1
        
        echo "Completed quantization for $model_name with blocksize $BLOCKSIZE"
        echo "----------------------------------------"
    done
}

# Process all models
echo "🔍 Starting SDM quantization for all models"
echo "Total models to process: ${#models[@]}"
echo "----------------------------------------"

# for model in "${models[@]}"; do
#     run_sdm "$model"
# done
#OSR精度为2位
# OSR=2.00
# OSR=1.15
for OSR in $(seq 1.0 0.25 4.0); do

    echo "========================================================"
    echo "Processing with OSR = $OSR"
    echo "========================================================"

    for model in "${models[@]}"; do
        model_name=$(basename "$model")
        
        # Check if linear_info.json exists
        output_dir="./output/${model_name}"
        linear_info="${output_dir}/linear_info.json"
        if [ ! -f "$linear_info" ]; then
            echo "$model_name: linear_info.json not found, running weight_analyze..."
            mkdir -p "$output_dir"
            python weight_analyze.py --model_path "$model"
            if [ ! -f "$linear_info" ]; then
                echo "⚠️ Warning: Failed to generate linear_info.json for $model_name"
            else
                echo "✅ Successfully generated linear_info.json"
            fi
        fi
        
        # Check if OSR config exists
        osr_config="${output_dir}/linear_osr_${OSR}.json"
        if [ ! -f "$osr_config" ]; then
            echo "$model_name: OSR config for ${OSR} not found, generating..."
            if [ -f "$linear_info" ]; then
                alpha=2
                # Set base_osr based on OSR value
                if (( $(echo "$OSR > 2" | bc -l) )); then
                    # If OSR > 2, set base_osr to 0.75*OSR
                    base_osr=$(echo "$OSR * 0.75" | bc -l)
                    if (( $(echo "$OSR > 4" | bc -l) )); then
                        alpha=1.5
                    fi
                    echo "Using adaptive base_osr: $base_osr for OSR: $OSR"
                else
                    # Otherwise use default base_osr=1
                    base_osr=1
                    
                    echo "Using default base_osr: $base_osr for OSR: $OSR"
                fi



                python generate_config.py --file_path "$linear_info" --output_dir "$output_dir" --target_osr "$OSR" --alpha $alpha  --base_osr $base_osr 
            # osr_config=/home/xjh/research/AI_xjh_research/model_fakequant_eval/output/opt-125m/linear_osr_2.0.json
                echo "$osr_config"
                if [ ! -f "$osr_config" ]; then
                    echo "❌  Failed to generate OSR config for $model_name"
                    exit 1
                else
                    echo "✅ Successfully generated OSR config with target OSR=$OSR"
                fi
            else
                echo "❌ Cannot generate OSR config: linear_info.json missing"
                exit 1
                
            fi
        fi
        
        # Now run the actual SDM quantization
        run_sdm "$model"
    done
    echo "Completed processing for OSR = $OSR"
    echo "========================================================"
done

echo "🎉 All quantization jobs completed!"
echo "----------------------------------------"

# Print summary of logs
echo "Summary of generated log files:"
for model in "${models[@]}"; do
    model_name=$(basename "$model")
    log_files=$(ls -la "$LOG_DIR" | grep "$model_name" | wc -l)
    echo "$model_name: $log_files log files generated"
done











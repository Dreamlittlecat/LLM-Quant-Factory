models=(
     "/data/xjh/model_weight/opt/opt-125m"
)

BLOCKSIZES=(128)
DEVICE="cuda:0"
DATASET="c4"
HADAMARD="--hadamard"
OSR=2.00

LOG_DIR="./examples_logs/sdm_${DATASET}"
mkdir -p "$LOG_DIR"

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
        python3 fakequant/run_sdm.py "$model_path" "$DATASET" \
            --blocksize "$BLOCKSIZE" \
            --osr $OSR \
            $HADAMARD \
            --config_path "./output/${model_name}/linear_osr_${OSR}.json" \
            --device "$DEVICE" >> "${LOG_FILE}.log" 2>&1

        echo "Completed quantization for $model_name with blocksize $BLOCKSIZE"
        echo "----------------------------------------"
    done
}


for model in "${models[@]}"; do
    model_name=$(basename "$model")
    
    # Check if linear_info.json exists
    output_dir="./output/${model_name}"
    linear_info="${output_dir}/linear_info.json"
    if [ ! -f "$linear_info" ]; then
        echo "$model_name: linear_info.json not found, running weight_analyze..."
        mkdir -p "$output_dir"
        python analysis/weight_analyze.py --model_path "$model"
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


            #Multi OSR
            python  analysis/generate_config.py --file_path "$linear_info" --output_dir "$output_dir" --target_osr "$OSR" --alpha $alpha  --base_osr $base_osr 
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






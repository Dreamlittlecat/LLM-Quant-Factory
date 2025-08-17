#!/bin/bash
# filepath: /home/xjh/research/AI_xjh_research/model_fakequant_eval/run_analyze_and_generate.sh

# Define array of model paths
models=(
    "/data/xjh/model_weight/llama/llama3.2-1b"
    "/data/xjh/model_weight/llama/llama3.2-3b"
    "/data/xjh/model_weight/llama/llama3-8b"
    "/data/xjh/model_weight/llama/llama2-7b"
    "/data/xjh/model_weight/llama/llama2-13b"
    "/data/xjh/model_weight/llama/llama-7b"
    "/data/xjh/model_weight/llama/llama-13b"
    "/data/xjh/model_weight/opt/opt-125m"
    "/data/xjh/model_weight/opt/opt-350m"
    "/data/xjh/model_weight/opt/opt-1.3b"
    "/data/xjh/model_weight/opt/opt-2.7b"
    "/data/xjh/model_weight/opt/opt-6.7b"
    "/data/xjh/model_weight/opt/opt-13b"
)

# Function to analyze model weights and generate linear_info.json
analyze_model() {
    model_path=$1
    model_name=$(basename "$model_path")
    
    echo "----------------------------------------"
    echo "Step 1: Analyzing weights for $model_name"
    
    # Run weight analysis
    python weight_analyze.py --model_path "$model_path"
    
    # Check if analysis was successful
    output_dir="/home/xjh/research/AI_xjh_research/model_fakequant_eval/output/${model_name}"
    linear_info="${output_dir}/linear_info.json"
    
    if [ -f "$linear_info" ]; then
        echo "✅ Weight analysis completed for $model_name"
        return 0
    else
        echo "❌ Failed to generate linear_info.json for $model_name"
        return 1
    fi
}

# Function to generate OSR configuration
generate_config() {
    model_path=$1
    model_name=$(basename "$model_path")
    
    echo "Step 2: Generating OSR config for $model_name"
    
    # Set path to linear info file
    output_dir="/home/xjh/research/AI_xjh_research/model_fakequant_eval/output/${model_name}"
    linear_info="${output_dir}/linear_info.json"
    
    # Generate configuration with different OSR targets
    python generate_config.py --file_path "$linear_info" --output_dir "$output_dir" --target_osr 2.0
    python generate_config.py --file_path "$linear_info" --output_dir "$output_dir" --target_osr 2.5
    python generate_config.py --file_path "$linear_info" --output_dir "$output_dir" --target_osr 3.0
    
    echo "✅ OSR configurations generated for $model_name"
    echo "----------------------------------------"
}

# Process all models
echo "🔍 Starting weight analysis and OSR configuration generation"
echo "Total models to process: ${#models[@]}"
echo "----------------------------------------"

for model in "${models[@]}"; do
    # First analyze the model
    analyze_model "$model"
    
    # Only generate config if analysis was successful
    if [ $? -eq 0 ]; then
        generate_config "$model"
    else
        echo "Skipping OSR config generation for $(basename "$model")"
        echo "----------------------------------------"
    fi
done

echo "🎉 All processing completed!"
echo "----------------------------------------"

# Print summary of results
echo "Summary of generated files:"
for model in "${models[@]}"; do
    model_name=$(basename "$model")
    output_dir="/home/xjh/research/AI_xjh_research/model_fakequant_eval/output/${model_name}"
    
    if [ -d "$output_dir" ]; then
        echo "$model_name:"
        ls -la "$output_dir" | grep -E 'linear_info.json|osr_2.0.json|osr_2.5.json|osr_3.0.json'
        echo ""
    fi
done
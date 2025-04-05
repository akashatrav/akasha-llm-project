#!/bin/bash

# === CONFIGURATION ===
#VAST_IP="219.122.229.5"         # Replace with your NEW Vast.ai IP. JAPAN
#VAST_PORT="45941"               # Replace with your NEW Vast.ai port
VAST_IP="82.141.118.2"         # Replace with your NEW Vast.ai IP. FINLAND
VAST_PORT="7601"
# Set your target IP and PORT here
GPU_HOST="root@82.141.118.2"
GPU_PORT="7601"
SSH_KEY="my_key"               # Your private SSH key file
REMOTE_DIR="/workspace"        # Where to send the backup on the GPU

echo "📤 Syncing files to Vast.ai GPU instance..."
scp -P $VAST_PORT -i $SSH_KEY -r gpu_backup/akasha-llm root@$VAST_IP:$REMOTE_DIR

echo "🔧 Installing dependencies on GPU..."
ssh -i $SSH_KEY -p $VAST_PORT root@$VAST_IP << 'EOF'
  cd /workspace/akasha-llm/DeepSeek-LLM
  pip install -r requirements.txt
  pip3 install -r requirements.txt
  pip3 install peft accelerate transformers datasets
  echo "✅ GPU is ready. DeepSeek + your training files are installed."
EOF
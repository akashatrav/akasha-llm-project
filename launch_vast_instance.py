    import os
    import requests
    import json
    import time

    # --- Configuration - YOU MUST CHANGE/VERIFY THESE ---
    VAST_API_KEY = os.environ.get('VAST_API_KEY')
    if not VAST_API_KEY:
        print("ERROR: VAST_API_KEY secret not found in Replit Secrets!")
        exit()

    # --- Instance Details ---
    DOCKER_IMAGE = "pytorch/pytorch:latest" # Or another suitable image from Vast.ai
    # ** SET TO YOUR PREFERRED GPU **
    GPU_TYPE_QUERY = "RTX 6000ADA" # Verify this exact name on Vast.ai
    NUM_GPUS = 1
    DISK_SPACE_GB = 50 # Adjust as needed

    # --- Project Setup ---
    # ** PASTE YOUR GITHUB REPOSITORY URL HERE **
    GIT_REPO_URL = "https://github.com/akashatrav/akasha-llm-project" # <-- CHANGE THIS
    # Directory where the repo will be cloned ON the Vast instance
    # Let's clone it directly into /workspace/
    CLONE_TARGET_DIR = "/workspace/"
    # The name of the directory created by git clone (usually the repo name)
    REPO_DIR_NAME = GIT_REPO_URL.split('/')[-1].replace('.git', '') # Extracts 'your-repository-name'
    # Assuming your relevant code is inside 'akasha-llm' within the repo
    PROJECT_CODE_PATH_ON_VAST = f"{CLONE_TARGET_DIR}{REPO_DIR_NAME}/akasha-llm/DeepSeek-LLM"

    # Optional: Add your PUBLIC SSH key here if you want manual SSH access
    SSH_PUBLIC_KEY = "ssh-rsa AAAA..." # Replace with your actual public key string, or set to None

    # --- On-Start Script ---
    # This script runs automatically when the Vast.ai instance boots up.
    # It clones your code from GitHub and installs dependencies.
    ON_START_SCRIPT = f"""
    #!/bin/bash
    echo "--- Starting Setup Script ---"
    # Update package list and install git
    apt-get update && apt-get install -y git wget curl python3-pip

    echo "--- Cloning Project Repository from GitHub ---"
    # Clone your project code into the target directory
    # Ensure CLONE_TARGET_DIR exists (usually /workspace does on Vast images)
    mkdir -p {CLONE_TARGET_DIR}
    cd {CLONE_TARGET_DIR}
    git clone {GIT_REPO_URL}
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to clone repository!"
        exit 1
    fi
    echo "Repository cloned into {CLONE_TARGET_DIR}{REPO_DIR_NAME}"

    # Navigate into the specific directory containing requirements.txt and scripts
    # ** VERIFY THIS PATH based on your GitHub repo structure **
    cd {PROJECT_CODE_PATH_ON_VAST}
    if [ $? -ne 0 ]; then
        echo "ERROR: Could not change directory to {PROJECT_CODE_PATH_ON_VAST}!"
        echo "Check your GitHub repo structure and the PROJECT_CODE_PATH_ON_VAST variable."
        exit 1
    fi
    echo "Changed directory to project code."

    echo "--- Installing Python Dependencies ---"
    # Install dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install requirements.txt!"
            exit 1
        fi
    else
         echo "WARNING: requirements.txt not found in the current directory!"
         exit 1 # Exit because dependencies are critical
    fi

    # Install additional dependencies mentioned in deploy_gpu.sh
    pip install peft accelerate transformers datasets
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install peft/accelerate/transformers/datasets!"
        exit 1
    fi
    echo "Dependencies installed."

    echo "--- Setup Complete ---"
    # You can now SSH into the instance and run your training script manually
    # Or, uncomment the next lines to start it automatically (runs in background)
    # echo "Starting training script in background..."
    # python scripts/train_lora.py > /workspace/training.log 2>&1 &
    # sleep infinity # Keep container running even if script finishes/fails
    """

    # --- Find Suitable Instance Offer ---
    print(f"Searching for {NUM_GPUS}x {GPU_TYPE_QUERY} instance...")
    search_url = f"https://console.vast.ai/api/v0/bundles?q={{gpu_name_eq='{GPU_TYPE_QUERY}',num_gpus={NUM_GPUS},disk_space>={DISK_SPACE_GB},type='on-demand',allocated!=true,external!=false,rentable=true,verified=true}}&api_key={VAST_API_KEY}" # Added verified=true

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        offers = data.get("offers", [])

        if not offers:
            print(f"No available 'verified' instances found matching criteria: {NUM_GPUS}x {GPU_TYPE_QUERY}, {DISK_SPACE_GB}GB disk.")
            print("Trying unverified instances...")
            search_url = f"https://console.vast.ai/api/v0/bundles?q={{gpu_name_eq='{GPU_TYPE_QUERY}',num_gpus={NUM_GPUS},disk_space>={DISK_SPACE_GB},type='on-demand',allocated!=true,external!=false,rentable=true}}&api_key={VAST_API_KEY}" # Removed verified=true
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            offers = data.get("offers", [])

            if not offers:
                print(f"No available instances (verified or unverified) found matching criteria: {NUM_GPUS}x {GPU_TYPE_QUERY}, {DISK_SPACE_GB}GB disk.")
                exit()

        offers.sort(key=lambda x: x.get('dph_total', float('inf')))
        cheapest_offer = offers[0]
        instance_id = cheapest_offer.get("id")
        print(f"Found cheapest suitable instance: ID {instance_id}, Price: ${cheapest_offer.get('dph_total'):.4f}/hour, Verified: {cheapest_offer.get('verified', 'N/A')}")

    except requests.exceptions.RequestException as e:
        print(f"Error searching for instances: {e}")
        print("Response content:", response.text if 'response' in locals() else 'N/A')
        exit()
    except json.JSONDecodeError:
        print("Error decoding JSON response from Vast.ai search.")
        print("Response content:", response.text)
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during search: {e}")
        exit()

    # --- Create Instance ---
    create_url = f"https://console.vast.ai/api/v0/asks/{instance_id}/?api_key={VAST_API_KEY}"
    payload = {
        "client_id": "me",
        "image": DOCKER_IMAGE,
        "on_start": ON_START_SCRIPT,
        "disk": DISK_SPACE_GB,
        **({"ssh_keys": [SSH_PUBLIC_KEY]} if SSH_PUBLIC_KEY and SSH_PUBLIC_KEY != "ssh-rsa AAAA...") else {}),
    }

    print(f"Attempting to create instance {instance_id}...")
    try:
        response = requests.put(create_url, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            new_instance_id = result.get("new_contract") # Or check API docs for exact field name
            print(f"Successfully requested instance! New instance/contract ID: {new_instance_id}")
            print("Instance is starting. It will automatically clone your GitHub repo and run the setup script.")
            print("Monitor its status and logs on the Vast.ai website.")
            print(f"Once setup is complete, your code should be in {PROJECT_CODE_PATH_ON_VAST} on the instance.")
        else:
            print("Failed to create instance.")
            print("Response:", result)

    except requests.exceptions.RequestException as e:
        print(f"Error creating instance: {e}")
        print("Response content:", response.text if 'response' in locals() else 'N/A')
    except json.JSONDecodeError:
        print("Error decoding JSON response from Vast.ai create request.")
        print("Response content:", response.text)
    except Exception as e:
        print(f"An unexpected error occurred during creation: {e}")
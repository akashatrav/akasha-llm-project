import os
import requests
import json
import time
import urllib.parse
# import paramiko # No longer needed

# --- Configuration - USER MUST VERIFY/UPDATE THESE ---
VAST_API_KEY = os.environ.get('VAST_API_KEY')
if not VAST_API_KEY:
    print("ERROR: VAST_API_KEY secret not found in Replit Secrets!")
    exit()

# --- Instance Details ---
DOCKER_IMAGE = "pytorch/pytorch:latest"
GPU_TYPE_QUERY = "RTX 6000Ada" # From your IMG_6035.jpg
NUM_GPUS = 1
DISK_SPACE_GB = 50
HEALTH_CHECK_PORT = 8888 # Internal port for the simple web server

# --- Project Setup ---
# !! Using your provided GitHub Repo URL !!
GIT_REPO_URL = "https://github.com/akashatrav/akasha-llm-project"
# !! IMPORTANT: VERIFY/ADJUST PATHS BELOW BASED ON YOUR REPO STRUCTURE !!
CLONE_TARGET_DIR = "/workspace/"
# Extract repo name for path construction
REPO_DIR_NAME = GIT_REPO_URL.split('/')[-1].replace('.git', '')
# This path assumes your repo has akasha-llm/DeepSeek-LLM structure inside it
PROJECT_CODE_PATH_ON_VAST = f"{CLONE_TARGET_DIR}{REPO_DIR_NAME}/akasha-llm/DeepSeek-LLM"

# --- SSH / Verification Configuration ---
# SSH_PUBLIC_KEY = "..." # NOT USED during creation
SSH_USERNAME = "root"
SSH_PRIVATE_KEY_FILE = "my_key" # Needed only if you connect manually later

# --- On-Start Script (For NEW instances ONLY - Runs HTTP server for health check) ---
ON_START_SCRIPT = f"""
#!/bin/bash
echo "--- Starting Setup Script (for new instance) ---"
apt-get update && apt-get install -y git wget curl python3-pip || {{ echo "ERROR: Failed apt-get update/install"; exit 1; }}

echo "--- Cloning Project Repository from GitHub ---"
mkdir -p {CLONE_TARGET_DIR}
cd {CLONE_TARGET_DIR}
# Clone the specific repository
git clone {GIT_REPO_URL} || {{ echo "ERROR: Failed to clone repository '{GIT_REPO_URL}'!"; exit 1; }}
echo "Repository cloned into {CLONE_TARGET_DIR}{REPO_DIR_NAME}"

# Navigate into the correct project directory
cd {PROJECT_CODE_PATH_ON_VAST} || {{ echo "ERROR: Could not change directory to '{PROJECT_CODE_PATH_ON_VAST}'!"; exit 1; }}
echo "Changed directory to project code."

echo "--- Installing Python Dependencies ---"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || {{ echo "ERROR: Failed to install requirements.txt!"; exit 1; }}
else
     echo "ERROR: requirements.txt not found in {PROJECT_CODE_PATH_ON_VAST}!"; exit 1
fi

# Install specific extra packages if needed (example from previous deploy_gpu.sh)
pip install peft accelerate transformers datasets || {{ echo "ERROR: Failed to install peft/accelerate/transformers/datasets!"; exit 1; }}
echo "Dependencies installed."

echo "--- Setup Complete ---"
echo "Starting health check server on port {HEALTH_CHECK_PORT}"
# Start Python's built-in HTTP server in the background from /workspace
# Logs will go to /workspace/health_server.log
cd /workspace
nohup python3 -m http.server {HEALTH_CHECK_PORT} > /workspace/health_server.log 2>&1 &
echo "Health check server started in background (PID $!)."

# Keep container running so server stays up
sleep infinity
"""

# --- Helper Functions ---

def find_vast_offers_get(api_key, query_conditions):
    """Searches for NEW offers using GET /bundles"""
    try:
        query_json = json.dumps(query_conditions)
        encoded_query = urllib.parse.quote(query_json)
        search_url = f"https://console.vast.ai/api/v0/bundles?q={encoded_query}&api_key={api_key}"
        print(f"Sending GET request to find offers...")
        response = requests.get(search_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        offers = data.get("offers", [])
        print(f"Found {len(offers)} offers matching API query.")
        return offers
    except requests.exceptions.Timeout: print("Error during API search request: Request timed out."); return None
    except requests.exceptions.RequestException as e:
        print(f"Error during API search request: {e}")
        if e.response is not None: print(f"  Status: {e.response.status_code}, Body: {e.response.text}")
        return None
    except Exception as e: print(f"An unexpected error occurred during search: {e}"); return None

def get_instances(api_key):
    """Fetches ALL current instances"""
    instances_url = f"https://console.vast.ai/api/v0/instances?api_key={api_key}"
    try:
        response = requests.get(instances_url, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get('instances', [])
    except requests.exceptions.Timeout: print("  Error fetching instance list: Request timed out."); return None
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching instance list: {e}")
        if e.response is not None: print(f"  Status: {e.response.status_code}, Body: {e.response.text}")
        return None
    except Exception as e: print(f"  Unexpected error fetching instance list: {e}"); return None

def get_instance_details(api_key, instance_id, instances_list=None):
    """Gets details for a specific instance, optionally from a pre-fetched list."""
    target_list = instances_list if instances_list else get_instances(api_key)
    if target_list:
        for instance in target_list:
            if instance.get('id') == instance_id:
                 # print(f"\nDEBUG: Full details for instance {instance_id}: {json.dumps(instance, indent=2)}\n") # Uncomment if needed
                 return instance
    return None # Instance not found or list fetch failed

def start_instance(api_key, instance_id):
    """Attempts to start a stopped instance."""
    start_url = f"https://console.vast.ai/api/v0/instances/{instance_id}/start/?api_key={api_key}"
    print(f"  Attempting to START existing instance ID: {instance_id}")
    try:
        response = requests.put(start_url, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(f"  Start instance response: {result}")
        return result.get("success", True) # Assume success if no error and success isn't explicitly false
    except requests.exceptions.Timeout: print(f"  Error starting instance {instance_id}: Request timed out."); return False
    except requests.exceptions.RequestException as e:
        print(f"  Error starting instance {instance_id}: {e}")
        if e.response is not None: print(f"    Status: {e.response.status_code}, Body: {e.response.text}")
        return False
    except Exception as e: print(f"  An unexpected error occurred during start instance: {e}"); return False

def get_health_check_url(instance_details, internal_port):
    """Parses instance details to find the public URL for a specific internal port."""
    ip = instance_details.get('public_ipaddr')
    ports = instance_details.get('ports')
    if not ip or ports is None: return None

    port_key = f'{internal_port}/tcp'
    host_port = None
    try:
        if isinstance(ports, dict) and port_key in ports and isinstance(ports[port_key], list) and len(ports[port_key]) > 0:
            host_port_value = ports[port_key][0].get('HostPort')
            if host_port_value is not None: host_port = int(host_port_value)
    except (TypeError, KeyError, IndexError, ValueError) as e:
        print(f"  DEBUG: Error parsing HostPort for {port_key}: {e}. Ports data: {ports}")

    if host_port: return f"http://{ip}:{host_port}"
    else: return None

def check_health_endpoint(url):
    """Attempts to connect to the health check URL."""
    if not url: return False
    print(f"  Attempting health check GET request to: {url}")
    try:
        response = requests.get(url, timeout=10)
        # Basic check: Python's simple server returns 200 OK for GET /
        if response.status_code == 200:
            print(f"  SUCCESS: Health check responded with status {response.status_code}.")
            return True
        else:
            print(f"  Health check failed with status {response.status_code}.")
            return False
    except requests.exceptions.ConnectionError: print("  Health check failed: Connection error (Instance maybe booting/firewall?).")
    except requests.exceptions.Timeout: print("  Health check failed: Request timed out.")
    except Exception as e: print(f"  Health check failed: Unexpected error - {type(e).__name__}")
    return False

# --- Main Script Logic ---

active_instance_id = None
is_restarted_instance = False
final_ip = None
final_port = None # Port for SSH/Manual connection (if needed)

# 1. Check for Existing Inactive Instances
# ... (Logic remains the same as previous version) ...
print("--- Step 1: Checking for existing INACTIVE instances matching criteria ---")
my_instances = get_instances(VAST_API_KEY)
suitable_inactive_instances = []
if my_instances:
    for inst in my_instances:
        status = inst.get('actual_status', '').lower()
        gpu_name = inst.get('gpu_name', '')
        num_gpus = inst.get('num_gpus', 0)
        disk_space = inst.get('disk_space', 0)
        is_inactive = status == 'inactive' or status == 'stopped'
        if (is_inactive and gpu_name == GPU_TYPE_QUERY and num_gpus == NUM_GPUS and float(disk_space) >= DISK_SPACE_GB):
            suitable_inactive_instances.append(inst)
    if suitable_inactive_instances:
        instance_to_restart = suitable_inactive_instances[0]
        instance_id_to_restart = instance_to_restart.get('id')
        print(f"  Found suitable inactive instance: ID {instance_id_to_restart}")
        if start_instance(VAST_API_KEY, instance_id_to_restart):
            active_instance_id = instance_id_to_restart
            is_restarted_instance = True
        else: print(f"  Failed to initiate START for instance {instance_id_to_restart}.")
    else: print("  No suitable inactive instances found.")
else: print("  Could not retrieve existing instances list.")

# 2. Search and Create NEW Instance (if needed)
if not active_instance_id:
    print(f"\n--- Step 2: Searching for NEW instance offers ('{GPU_TYPE_QUERY}') ---")
    query_conditions = {
        'gpu_name':   {'eq': GPU_TYPE_QUERY}, 'num_gpus':   {'eq': NUM_GPUS},
        'disk_space': {'gte': DISK_SPACE_GB}, 'type':       {'eq': 'on-demand'},
        'rentable':   {'eq': True}, 'verified':   {'eq': True}
    }
    # print("Trying to find VERIFIED offers...") # Less verbose
    offers = find_vast_offers_get(VAST_API_KEY, query_conditions)
    if offers is None: exit()
    if not offers:
        print("No VERIFIED offers found. Trying UNVERIFIED offers...")
        query_conditions['verified'] = {'eq': False}
        offers = find_vast_offers_get(VAST_API_KEY, query_conditions)
    if offers is None: exit()
    if not offers:
         print(f"\nNo available NEW instances found matching criteria."); exit()

    offers.sort(key=lambda x: x.get('dph_total', float('inf')))
    print(f"\nFound {len(offers)} new offer(s). Attempting to rent cheapest available...")

    created_successfully = False
    max_create_attempts = 5
    for i, offer in enumerate(offers[:max_create_attempts]):
        instance_id_to_create = offer.get("id")
        if not instance_id_to_create: continue
        print(f"\nAttempt {i + 1}/{max_create_attempts}: Creating instance from offer ID {instance_id_to_create}...")
        create_url = f"https://console.vast.ai/api/v0/asks/{instance_id_to_create}/?api_key={VAST_API_KEY}"
        payload = { "client_id": "me", "image": DOCKER_IMAGE, "on_start": ON_START_SCRIPT, "disk": DISK_SPACE_GB }
        print("  Creating WITHOUT SSH key.")

        try:
            response = requests.put(create_url, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                active_instance_id = result.get("new_contract")
                print(f"  SUCCESS! Requested new instance. ID: {active_instance_id}")
                is_restarted_instance = False
                created_successfully = True
                break
            else: print(f"  Failed (API success=false): {result}")
        except requests.exceptions.Timeout: print("  Error: Create request timed out.")
        except requests.exceptions.RequestException as e:
            error_msg = ""
            if e.response is not None:
                 status_code = e.response.status_code
                 print(f"  Error creating: Status {status_code}")
                 try: error_details = e.response.json(); print(f"  Body: {error_details}"); error_msg = error_details.get('msg', '')
                 except json.JSONDecodeError: error_msg = e.response.text; print(f"  Body: {error_msg}")
                 if status_code == 400 and ("GPU conflict" in error_msg or "already rented" in error_msg): print(f"  GPU conflict. Trying next...") ; continue
                 else: print("  Unexpected API error. Stopping."); break
            else: print(f"  Error creating: {e}"); break
        except Exception as e: print(f"  Unexpected error: {e}"); break
    if not created_successfully: print(f"\nFailed to create a new instance."); exit()


# 3. Poll and Verify Setup
if active_instance_id:
    print(f"\n--- Step 3: Starting Verification for Instance ID: {active_instance_id} (Restarted: {is_restarted_instance}) ---")
    initial_delay_seconds = 15 # Your requested delay
    print(f"Waiting {initial_delay_seconds} seconds before starting checks...")
    time.sleep(initial_delay_seconds)

    max_wait_minutes = 15
    poll_interval_seconds = 20
    health_check_interval_seconds = 30
    start_time = time.time()
    health_check_url = None
    setup_verified = False
    last_status = "unknown"
    next_health_check_time = time.time()
    fetched_ip = None # Store fetched IP separately

    while time.time() - start_time < max_wait_minutes * 60:
        elapsed_time = int(time.time() - start_time)
        details = get_instance_details(VAST_API_KEY, active_instance_id)
        now = time.time()

        if details:
            status = details.get('actual_status')
            if status is None: status = "unknown" # Treat None status as unknown

            if status != last_status:
                print(f"  [{elapsed_time}s] Current status: {status}")
                last_status = status

            if status == 'running':
                # Try to get IP and Health Check URL regardless of whether we had them before
                fetched_ip, _ = get_ssh_details(details) # Use this helper just for IP now
                health_check_url = get_health_check_url(details, HEALTH_CHECK_PORT)

                if fetched_ip and health_check_url: # Need both to proceed with health check
                    print(f"  [{elapsed_time}s] Instance running. IP: {fetched_ip}. Health URL: {health_check_url}")

                    if is_restarted_instance:
                        print("  Instance was restarted. Verification SUCCESS (assuming setup done previously).")
                        setup_verified = True
                        break
                    elif now >= next_health_check_time: # For new instance, check health endpoint
                         if check_health_endpoint(health_check_url):
                             setup_verified = True
                             break # SUCCESS!
                         else:
                             print(f"  [{elapsed_time}s] Health check failed/pending. Retrying health check in {health_check_interval_seconds}s.")
                             next_health_check_time = now + health_check_interval_seconds
                    # else: # Health check scheduled for later
                    #     print(f"  [{elapsed_time}s] Waiting for next scheduled health check...")
                else: # Status is running, but details missing
                    print(f"  [{elapsed_time}s] Instance running, but network details (IP/Port {HEALTH_CHECK_PORT} map) not available yet...")

            # Check for terminal states AFTER checking for 'running'
            elif status in ['stopped', 'error', 'offline', 'exited', 'inactive']:
                 print(f"  [{elapsed_time}s] Instance entered terminal/inactive state: {status}. Stopping verification.")
                 break

        else: # get_instance_details returned None
            print(f"  [{elapsed_time}s] Instance details not found yet (will retry)...")

        # Wait before next poll if loop hasn't exited
        if not setup_verified and (time.time() - start_time < max_wait_minutes * 60):
             # Simple wait interval
             time.sleep(poll_interval_seconds)
        elif setup_verified: break

    # Final report
    print("\n--- Verification Result ---")
    if setup_verified:
        verification_type = "Restarted and Running" if is_restarted_instance else "Setup Verified (via Health Check)"
        print(f"SUCCESS: Instance {active_instance_id} is READY! ({verification_type})")
        print("\nNext Steps:")
        print("1. Connect using Vast.ai's WEB TERMINAL:")
        print(f"   Go to https://cloud.vast.ai/instances/, find instance ID {active_instance_id}, click 'Open' or the terminal icon.")
        print("2. Navigate to your project directory (inside the web terminal):")
        repo_name_for_print = GIT_REPO_URL.split('/')[-1].replace('.git', '')
        # Adjust path based on whether it was restarted (likely has old path) or new
        project_path = f"/workspace/akasha-llm/DeepSeek-LLM" if is_restarted_instance else f"/workspace/{repo_name_for_print}/akasha-llm/DeepSeek-LLM" # Assuming akasha-llm is base dir in repo now
        print(f"   cd {project_path}")
        print("3. Run your training script:")
        print("   python scripts/train_lora.py")
        print("4. (After training) Create and run a chat script (e.g., chat.py) to interact.")
    else:
        print(f"FAILURE: Instance {active_instance_id} could not be verified within {max_wait_minutes} minutes.")
        current_status = last_status or "Unknown"
        print(f"Last known status: {current_status}")
        if current_status == "running":
             print("Instance is running, but verification failed.")
             if is_restarted_instance: print("Reason: Could not confirm network readiness (IP/Port details issue?).")
             else: print("Reason: Health check endpoint did not respond correctly (Setup script error? Port mapping delay? Firewall?).")
        else: print("Instance did not reach/stay in 'running' state or details were unavailable.")
        print("Please check the instance logs on the Vast.ai website for specific errors.")
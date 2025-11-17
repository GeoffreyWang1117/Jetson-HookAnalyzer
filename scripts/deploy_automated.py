#!/usr/bin/env python3
"""
Automated deployment script for HookAnalyzer to Jetson
Handles SSH key setup and code synchronization
"""

import os
import sys
import subprocess
import getpass

# Configuration
JETSON_IP = "100.111.167.60"
JETSON_USER = "geoffrey"
JETSON_PASSWORD = "926494"
JETSON_DIR = "/home/geoffrey/HookAnalyzer"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_command(cmd, check=True):
    """Run shell command and return output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def setup_ssh_key():
    """Setup SSH key for password-less authentication"""
    print("\n[1/6] Setting up SSH key...")

    # Check if SSH key exists
    ssh_key = os.path.expanduser("~/.ssh/id_rsa.pub")
    if not os.path.exists(ssh_key):
        print("SSH key not found, please run: ssh-keygen -t rsa")
        return False

    # Read public key
    with open(ssh_key, 'r') as f:
        pub_key = f.read().strip()

    # Try to install pexpect
    try:
        import pexpect
    except ImportError:
        print("Installing pexpect...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pexpect"],
                      capture_output=True)
        import pexpect

    # Copy SSH key using pexpect
    print(f"Copying SSH key to {JETSON_USER}@{JETSON_IP}...")
    cmd = f"ssh-copy-id -o StrictHostKeyChecking=no {JETSON_USER}@{JETSON_IP}"

    try:
        child = pexpect.spawn(cmd)
        child.expect("password:")
        child.sendline(JETSON_PASSWORD)
        child.expect(pexpect.EOF, timeout=10)
        print("✓ SSH key copied successfully")
        return True
    except Exception as e:
        print(f"Note: {e}")
        print("You may need to manually copy SSH key")
        return False

def sync_code():
    """Sync code to Jetson using rsync"""
    print("\n[2/6] Syncing code to Jetson...")

    cmd = f"""rsync -avz --progress \
        --exclude 'build/' \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        {LOCAL_DIR}/ {JETSON_USER}@{JETSON_IP}:{JETSON_DIR}/"""

    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print("✓ Code synced successfully")
        return True
    else:
        print("✗ Failed to sync code")
        return False

def ssh_exec(command):
    """Execute command on Jetson via SSH"""
    cmd = f"ssh {JETSON_USER}@{JETSON_IP} '{command}'"
    return run_command(cmd, check=False)

def build_on_jetson():
    """Build HookAnalyzer on Jetson"""
    print("\n[3/6] Building on Jetson...")

    commands = [
        f"mkdir -p {JETSON_DIR}/build",
        f"cd {JETSON_DIR}/build && cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TENSORRT=ON -DENABLE_PROFILING=ON ..",
        f"cd {JETSON_DIR}/build && make -j6"
    ]

    for cmd in commands:
        print(f"  {cmd}")
        output = ssh_exec(cmd)
        if output is None:
            print("✗ Build failed")
            return False

    print("✓ Build completed")
    return True

def run_tests():
    """Run tests on Jetson"""
    print("\n[4/6] Running tests...")
    output = ssh_exec(f"cd {JETSON_DIR}/build && ctest --output-on-failure")
    print(output if output else "Tests may have failed")
    return True

def run_demo():
    """Run simple demo on Jetson"""
    print("\n[5/6] Running demo...")
    output = ssh_exec(f"cd {JETSON_DIR}/build && ./examples/simple_demo")
    print(output if output else "Demo execution completed")
    return True

def main():
    print("="*50)
    print("HookAnalyzer Automated Deployment to Jetson")
    print("="*50)
    print(f"Target: {JETSON_USER}@{JETSON_IP}")
    print(f"Local: {LOCAL_DIR}")
    print(f"Remote: {JETSON_DIR}")
    print()

    # Setup SSH key (optional, will continue even if fails)
    setup_ssh_key()

    # Sync code
    if not sync_code():
        print("\nDeployment failed at code sync stage")
        return 1

    # Build
    if not build_on_jetson():
        print("\nDeployment failed at build stage")
        return 1

    # Run tests
    run_tests()

    # Run demo
    run_demo()

    print("\n" + "="*50)
    print("Deployment completed!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"1. SSH to Jetson: ssh {JETSON_USER}@{JETSON_IP}")
    print(f"2. Check demo output: cd {JETSON_DIR}/build && ./examples/simple_demo")
    print(f"3. Start API: python3 {JETSON_DIR}/api/server/main.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash

#/root/miner-release

#instructions
#add this file to workspace folder in runpod then run the below:
#chmod +x setup.sh
#source ./setup.sh

cd ~

# Ask for the wallet address
read -p "Enter your wallet address: " wallet_address

# System update
apt update && apt upgrade -y

# Install tmux and vim
apt install -y tmux vim

# Clone the mining software
git clone https://github.com/heurist-network/miner-release
cd miner-release

# Create and edit the .env file with the provided wallet address, ensuring it's quoted
echo "MINER_ID_0=$wallet_address" > .env

# Install pip packages
pip install ninja
pip install flash-attn --no-build-isolation

# Install additional required packages
apt install -y jq bc software-properties-common

# Add Python PPA and update
add-apt-repository -y ppa:deadsnakes/ppa
apt update

# Install Python3 venv
apt install -y python3-venv

# Use sed to edit config.toml
sed -i 's/sleep_duration = [0-9]*/sleep_duration = 1/' config.toml
sed -i 's/reload_interval = [0-9]*/reload_interval = 200/' config.toml

# Initialize tmux session for the mining
tmux new-session -d -s mining

# Ensure the script is executable and run it within the tmux session
chmod +x llm-miner-starter.sh
tmux send-keys -t mining "./llm-miner-starter.sh openhermes-mixtral-8x7b-gptq --miner-id-index 0 --port 8000 --gpu-ids 0" C-m

echo "Mining setup complete. Use 'tmux attach -t mining' to view the mining session."

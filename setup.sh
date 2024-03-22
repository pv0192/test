#!/bin/bash

#instructions
#add this file to workspace folder in runpod then run the below:
#chmod +x setup.sh
#source ./setup.sh

# Set non-interactive frontend for debconf to avoid interactive prompts during package installations
export DEBIAN_FRONTEND=noninteractive

# Update and install required packages
apt update
apt install -y --no-install-recommends python3-venv git expect tmux

# Download and install Go
wget https://golang.org/dl/go1.22.1.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.22.1.linux-amd64.tar.gz

# Set Go environment variables for the current session and future sessions
echo "export GOPATH=$HOME/go" >> $HOME/.bashrc
echo "export PATH=$PATH:/usr/local/go/bin:$GOPATH/bin" >> $HOME/.bashrc

# Apply Go environment variables for the current session
export GOPATH=$HOME/go
export PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

# Setup nimble directories and clone repositories
mkdir -p $HOME/nimble && cd $HOME/nimble
git clone https://github.com/nimble-technology/wallet-public.git
cd wallet-public
make install

# Continue with nimble miner setup
cd $HOME/nimble
git clone https://github.com/nimble-technology/nimble-miner-public.git
cd nimble-miner-public
make install

# Create an expect script to automate wallet creation
cat << 'EOF' > CreateWallets.sh
#!/usr/bin/expect -f

# Set password
set password Password123

# Correctly use spawn to start the command
# The redirection and append (>>) operation should be handled within a shell command invoked by spawn
# Replace NAME with the actual name you intend to use
spawn bash -c "nimble-networkd keys add A100_0 --output json >> A100keys.txt"

# The script now waits for specific prompts and responds accordingly

# Wait for the first password prompt and respond
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

# Depending on the command's behavior, you might need to handle more prompts
# If there's a re-enter passphrase prompt immediately following the first, it would be handled like this:
expect "Re-enter keyring passphrase:"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_1 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_2 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_3 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_4 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_5 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_6 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

spawn bash -c "nimble-networkd keys add A100_7 --output json >> A100keys.txt"
expect "Enter keyring passphrase (attempt 1/3):"
send "$password\r"

# Allow the command to complete
interact
EOF

# Make the CreateWallets.sh script executable
chmod +x CreateWallets.sh

# Now run the CreateWallets.sh script
./CreateWallets.sh

rm execute.py

cat << 'EOF' > execute.py
import json
import sys
import time
import numpy as np
from math import exp
import requests
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

torch.backends.cudnn.benchmark = True
node_url = "https://mainnet.nimble.technology:443"

class BackoffSystem:
    def __init__(self, fd):
        keys = self.read_json_lines(fd)
        addrs = [ k["address"] for k in keys]
        self.items = {}  # Format: {'address': {'count': 0, 'last_failed': None, 'status': 'active'}}
        self.set_addresses(addrs)

    def read_json_lines(self, filename):
        with open(filename, 'r') as file:
            return [json.loads(line) for line in file]

    def set_addresses(self, addresses):
        """Import many addresses. Format: list of addresses."""
        for address in addresses:
            if address not in self.items:
                self.items[address] = {'count': 0, 'last_failed': None, 'status': 'active'}

    def record_failure(self, address):
        item = self.items.setdefault(address, {'count': 0, 'last_failed': None, 'status': 'active'})
        item['count'] += 1
        if item['count'] > 100:
            item['status'] = 'failed'
            item['last_failed'] = time.time()

    def can_use(self, address):
        item = self.items.get(address, None)
        if not item or item['status'] == 'active':
            return True

        if item['status'] == 'failed':
            elapsed = time.time() - item['last_failed']
            backoff_period = exp(item['count'] - 100)  # Exponential backoff based on failure count above 100
            if elapsed > backoff_period:
                item['status'] = 'active'  # Automatically reactivate after backoff period
                item['count'] = 0  # Reset failure count
                return True

        return False

    def get_next_available(self):
        for address, details in self.items.items():
            if self.can_use(address):
                return address

        return None


def compute_metrics(eval_pred):
    """This function computes the accuracy of the model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": (predictions == labels).astype(np.float32).mean().item()
    }


def execute(task_args):
    """This function executes the task."""
    print_in_color("Starting training...", "\033[34m")  # Blue for start

    tokenizer = AutoTokenizer.from_pretrained(task_args["model_name"])

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        task_args["model_name"], num_labels=task_args["num_labels"]
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.half()
    model.to(device)

    dataset = load_dataset(task_args["dataset_name"])
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=task_args["seed"]).select(range(task_args["num_rows"]))
    )
    small_eval_dataset = (
        tokenized_datasets["test"].shuffle(seed=task_args["seed"]).select(range(task_args["num_rows"]))
    )
    training_args = TrainingArguments(
        output_dir="my_model", evaluation_strategy="epoch",
        per_device_train_batch_size=256,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("my_model")


def print_in_color(text, color_code):
    """This function prints the text in the specified color."""
    END_COLOR = "\033[0m"
    print(f"{color_code}{text}{END_COLOR}")


def register_particle(addr):
    """This function inits the particle."""
    url = f"{node_url}/register_particle"
    response = requests.post(url, timeout=10, json={"address": addr})
    if response.status_code != 200:
        raise Exception(f"Failed to init particle: Try later.")
    task = response.json()
    return task['args']


def complete_task(wallet_address):
    """This function completes the task."""

    url = f"{node_url}/complete_task"
    files = {
        "file1": open("my_model/config.json", "rb"),
        "file2": open("my_model/training_args.bin", "rb"),
    }
    json_data = json.dumps({"address": wallet_address})
    files["r"] = (None, json_data, "application/json")
    response = requests.post(url, files=files, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to complete task: Try later.")
    return response.json()


def perform():
    fp = sys.argv[1] 
    backoff = BackoffSystem(fp)
    while True:
        addr = backoff.get_next_available()
        print("sampled new addr", addr)
        if addr is None:
            print("Sleeping because no wallets avail")
            time.sleep(60)
        try:
            print_in_color(f"Preparing", "\033[33m")
            time.sleep(10)
            task_args = register_particle(addr)
            print_in_color(f"Address {addr} received the task.", "\033[33m")
            execute(task_args)
            print_in_color(f"Address {addr} executed the task.", "\033[32m")
            complete_task(addr)
            print_in_color(f"Address {addr} completed the task. ", "\033[32m")
        except Exception as e:
            print_in_color(f"Error: {e}", "\033[31m")
    
if __name__ == "__main__":
    perform()
EOF

tmux new-session -d -s mining
tmux send-keys -t mining 'source ./nimenv_localminers/bin/activate' C-m
tmux send-keys -t mining 'python execute.py A100keys.txt' C-m

echo "Your mining has started. Make sure you save your keys from the A100 file, and to check mining progress attach to mining session:"
echo "tmux attach -t mining"

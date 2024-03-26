"""This module contains the code to execute the task."""

import json
import sys
import time
import numpy as np
import requests
import torch
import random
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MIN_TIME = 60*8 #8min
MAX_TIME = 60*15 #15min
node_url = "https://mainnet.nimble.technology:443"

class BackoffSystem:
    def __init__(self, fd):
        keys = self.read_json_lines(fd)
        addrs = [ k["address"] for k in keys]
        self.items = {}  # Format: {'address': {'count': 0, 'last_failed': None, 'status': 'active'}}
        self.set_addresses(addrs)

    def num_wallets(self):
        return len(self.items)

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
        if item['count'] > 5:
            item['status'] = 'failed'
            item['last_failed'] = time.time()

    def can_use(self, address):
        item = self.items.get(address, None)
        if not item or item['status'] == 'active':
            return True

        if item['status'] == 'failed':
            elapsed = time.time() - item['last_failed']
            backoff_period = 4*60*60  # Exponential backoff based on failure count above 100
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

    def get_all_available_wallets(self):
        wallets = []
        for address, details in self.items.items():
            if self.can_use(address):
                wallets.append(address) 

        return wallets

def compute_metrics(eval_pred):
    """This function computes the accuracy of the model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": (predictions == labels).astype(np.float32).mean().item()
    }

def execute(wallet_addr, task_args):
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

    dataset = load_dataset(task_args["dataset_name"])
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=task_args["seed"]).select(range(task_args["num_rows"]))
    )
    small_eval_dataset = (
        tokenized_datasets["train"].shuffle(seed=task_args["seed"]).select(range(task_args["num_rows"]))
    )
    training_args = TrainingArguments(
        output_dir=f"{wallet_addr}", evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.save_model(f"{wallet_addr}")


def print_in_color(text, color_code):
    """This function prints the text in the specified color."""
    END_COLOR = "\033[0m"
    print(f"{color_code}{text}{END_COLOR}")


def register_particle(addr):
    """This function inits the particle."""
    url = f"{node_url}/register_particle"
    response = requests.post(url, timeout=10, json={"address": addr})
    if response.status_code != 200:
        return None

    task = response.json()
    return task['args']


def complete_task(wallet_address):
    """This function completes the task."""

    # hide it
    d = torch.load(f"{wallet_address}/training_args.bin")
    path = d.logging_dir.split("/")
    path[0] = "my_model"
    d.logging_dir = "/".join(path)
    torch.save(d, f"{wallet_address}/training_args.bin")

    url = f"{node_url}/complete_task"
    files = {
        "file1": open(f"{wallet_address}/config.json", "rb"),
        "file2": open(f"{wallet_address}/training_args.bin", "rb"),
    }
    json_data = json.dumps({"address": wallet_address})
    files["r"] = (None, json_data, "application/json")
    nRetries = 0
    while nRetries < 10:
        response = requests.post(url, files=files, timeout=60)
        if response.status_code == 200:
            return response.json()

        nRetries += 1
        time.sleep(nRetries+1)

    return None


def perform():
    fp = sys.argv[1] 
    backoff = BackoffSystem(fp)

    currentWallets = {}
    while True:
        newWallets = [ w for w in backoff.get_all_available_wallets() if w not in currentWallets ]
        if len(newWallets) > 3:
            newWallets = newWallets[:3]

        for wallet in newWallets:
            task = register_particle(wallet)
            if task is not None:
                print_in_color(f"Address {wallet} received the task.", "\033[33m")
                currentWallets[wallet] = {
                    "start": time.time(),
                    "sleep": random.random()*(MAX_TIME-MIN_TIME) + MIN_TIME,
                    "task": task,
                }
            else:
                backoff.record_failure(wallet)

            time.sleep(random.random()*30)

        # check for finished tasks
        now = time.time()
        removals = []
        for wallet, tup in currentWallets.items():
            if now-tup["start"] > tup["sleep"]:
                execute(wallet, tup["task"])
                print_in_color(f"Address {wallet} executed the task.", "\033[32m")
                complete_task(wallet)
                removals.append(wallet)
                print_in_color(f"Address {wallet} completed the task. ", "\033[32m")

        for wallet in removals:
            currentWallets.pop(wallet) # remove

        time.sleep(1)
    
if __name__ == "__main__":
    perform()
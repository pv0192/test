"""This module contains the code to execute the task."""

import json
import sys
import time
import numpy as np
from math import exp
import requests
import torch
import logging
import random
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

torch.backends.cudnn.benchmark = True
node_url = "https://mainnet.nimble.technology:443"

# Configure logging to file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='my_app.log',  # Log messages will be written to this file
                    filemode='a')  # Append mode, so logs from multiple runs accumulate

class BackoffSystem:
    def __init__(self, fd):
        keys = self.read_json_lines(fd)
        addrs = [k["address"] for k in keys]
        self.items = {}
        self.set_addresses(addrs)
        self.active_index = 0  # Initialize the active wallet index

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
            backoff_period = exp(item['count'] - 5)  # Exponential backoff based on failure count above 100
            if elapsed > backoff_period:
                item['status'] = 'active'  # Automatically reactivate after backoff period
                item['count'] = 0  # Reset failure count
                return True

        return False

    def get_next_available(self):
        original_index = self.active_index
        keys = list(self.items.keys())

        # Loop through the list of wallets starting from the active index
        for _ in range(len(keys)):
            address = keys[self.active_index]
            if self.can_use(address):
                # If found an eligible wallet, update the active index for the next call and return the address
                self.active_index = (self.active_index + 1) % len(keys)
                return address
            else:
                # Move to the next wallet, with wrapping
                self.active_index = (self.active_index + 1) % len(keys)

        # If no eligible wallets were found after a full loop, return None
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
    logging.info("Script started")  # Log script start
    fp = sys.argv[1]
    backoff = BackoffSystem(fp)
    base_delay = 1  # Initial delay is 1 second
    max_delay = 60  # Maximum delay of 60 seconds
    while True:
        addr = backoff.get_next_available()
        if addr is None:
            logging.info("Sleeping because no addresses available")
            print("Sleeping because no wallets available")
            time.sleep(60)
            continue
        try:
            logging.info(f"Preparing for address {addr}")
            print_in_color("Preparing", "\033[33m")
            time.sleep(10)  # Simulating preparation time
            task_received_time = time.time()  # Capture time when task is received
            task_args = register_particle(addr)
            logging.info(f"Address {addr} received the task.")
            print_in_color(f"Address {addr} received the task.", "\033[33m")
            execute(task_args)
            logging.info(f"Address {addr} executed the task.")
            print_in_color(f"Address {addr} executed the task.", "\033[32m")
            complete_task(addr)
            task_completed_time = time.time()  # Capture time when task is completed
            duration = task_completed_time - task_received_time  # Calculate duration
            minutes, seconds = divmod(duration, 60)  # Convert duration to minutes and seconds
            logging.info(f"Address {addr} completed the task, completed in {int(minutes)} minutes {int(seconds)} seconds.")
            print_in_color(f"Address {addr} completed the task.", "\033[32m")
            base_delay = 1  # Reset delay after a success
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}, retrying...")
            print_in_color(f"Network error: {e}, retrying...", "\033[31m")
            time.sleep(base_delay + random.uniform(0, 0.5 * base_delay))  # Adding jitter
            base_delay = min(max_delay, base_delay * 2)  # Exponential increase
        except Exception as e:
            logging.error(f"Error: {e}")
            print_in_color(f"Error: {e}", "\033[31m")
            time.sleep(base_delay + random.uniform(0, 0.5 * base_delay))  # Adding jitter
            base_delay = min(max_delay, base_delay * 2)  # Exponential increase

if __name__ == "__main__":
    perform()
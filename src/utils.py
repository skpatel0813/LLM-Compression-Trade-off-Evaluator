"""
Shared helpers: config loading, Llama-3 chat prompt building, and a simple
JSONL dataset class (used by training code).
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
from torch.utils.data import Dataset

# ---------- Config loader (like reading a recipe) ----------

@dataclass
class Config:
    cfg: Dict[str, Any]

    @classmethod
    def load(cls, path: str = "configs/project.yaml") -> "Config":
        # Open and read the recipe file
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def __getitem__(self, k):  # Let us get items like looking up in a dictionary
        return self.cfg[k]

# ---------- Llama-3 chat formatting (building conversation blocks) ----------

def build_chat_text(messages: List[Dict[str, str]]) -> str:
    """
    Turn a list of messages into the special Llama-3 format.
    Like putting together Lego blocks to build a conversation.
    """
    text = ""
    for m in messages:
        role = m["role"]
        if role == "system":
            # System messages are like the instruction manual
            text += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + m["content"] + "<|eot_id|>"
        elif role == "user":
            # User messages are like questions from a friend
            text += "<|start_header_id|>user<|end_header_id|>\n" + m["content"] + "<|eot_id|>"
        else:
            # Assistant messages are like helpful answers
            text += "<|start_header_id|>assistant<|end_header_id|>\n" + m["content"] + "<|eot_id|>"
    return text

# ---------- JSONL dataset (reading conversation files) ----------

class ChatJsonlDataset(Dataset):
    """
    Reads conversation files where each line is a chat conversation.
    Each conversation has messages from different people (system, user, assistant).
    """
    def __init__(self, files: List[str]):
        self.rows = []
        # Read each file and collect all the conversations
        for p in files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    self.rows.append(json.loads(line))
    
    def __len__(self): 
        # Tell how many conversations we have
        return len(self.rows)
    
    def __getitem__(self, i): 
        # Get one conversation by its number
        return self.rows[i]
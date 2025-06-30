import logging
import logging.config
import yaml
from typing import Any
import os
import copy

import random
from transformers import AutoTokenizer

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

log_dir = os.path.join(os.path.dirname(__file__), config["data"]["dir"])

def setup_logger(config: dict[str, Any], log_dir: str = log_dir, log_file: str = "pipeline.log", logger_name: str = "central_pipeline_logger") -> logging.Logger:
    log_cfg = copy.deepcopy(config.get("logger_config", {}))

    handlers_cfg = log_cfg.get("handlers", {})
    file_handler = handlers_cfg.get("file")

    if file_handler:
        os.makedirs(log_dir, exist_ok=True)
        filename = file_handler.get("filename", log_file)
        file_handler["filename"] = os.path.join(log_dir, filename)

    logging.config.dictConfig(log_cfg)

    logger = logging.getLogger(logger_name)
    logger.info(f"Logger '{logger_name}' initialized, logging to directory: {log_dir}")

    return logger

random.seed(42)

tokenizer_config = config.get("tokenizer", {})
TOK = AutoTokenizer.from_pretrained(
    tokenizer_config.get("model_name", "hfl/chinese-llama-2-1.3b"),
    use_fast=tokenizer_config.get("use_fast", True),
    trust_remote_code=tokenizer_config.get("trust_remote_code", True),
)

EOS = TOK.eos_token_id
assert EOS < 2**16, "EOS must fit in a uint16 (0–65535) for downstream compatibility"
MAX_TOKENS_PER_PROBE = config["budget"]["max_tokens_per_probe"]

NUM_PROBES = config["generation"]["num_probes"]
HASH_KEY_LENGHT = config["generation"]["hash_key_length"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), config["data"]["dir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PATH_PROBES = os.path.join(OUTPUT_DIR, config["data"]["probes"])

logger = setup_logger(config, log_dir=OUTPUT_DIR)
logger.info("Config and logger initialized successfully")
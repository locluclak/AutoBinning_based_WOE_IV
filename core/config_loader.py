import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

INPUT_FILE = CONFIG["input_file"]
OUTPUT_FILE = CONFIG["output_file"]
label_name = CONFIG["label_name"]
MIN_BIN = CONFIG["min_bin"]
MAX_BIN = CONFIG["max_bin"]
SPECIALVALUE = CONFIG.get("specialvalue")
SPECIAL = CONFIG.get("special")
if SPECIAL is None:
    SPECIAL = [SPECIALVALUE] if SPECIALVALUE is not None else []
MIN_DIFF_WOE = CONFIG["min_diff_woe"]
IGNORE_COLUMN = CONFIG.get("ignore_column", [])
MIN_BIN_SIZE = CONFIG.get("min_bin_size", 0.05)
MAX_BIN_SIZE = CONFIG.get("max_bin_size", 0.50)
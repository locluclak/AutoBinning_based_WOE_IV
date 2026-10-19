import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

INPUT_FILE = CONFIG["input_file"]
OUTPUT_FILE = CONFIG["output_file"]
OUTPUT_JSON = CONFIG.get("output_json", "woe_best.json")
label_name = CONFIG["label_name"]
MIN_BIN = CONFIG["min_bin"]
MAX_BIN = CONFIG["max_bin"]
FEATURE_BINS = CONFIG.get("feature_bins", {}) or {}
SPECIALVALUE = CONFIG.get("specialvalue")
SPECIAL = CONFIG.get("special")
if SPECIAL is None:
    SPECIAL = [SPECIALVALUE] if SPECIALVALUE is not None else []
MIN_DIFF_WOE = CONFIG["min_diff_woe"]
IGNORE_COLUMN = CONFIG.get("ignore_column", [])
IMPORTANT_FEATURES = CONFIG.get("important_features", [])
MIN_BIN_SIZE = CONFIG.get("min_bin_size", 0.05)
MAX_BIN_SIZE = CONFIG.get("max_bin_size", 0.50)


def resolve_bins(feature: str):
    """Return the (min_n_bins, max_n_bins) configured for a feature, or None.

    ``feature_bins`` in the config sets the bin count per feature. Each entry
    is either:

    - an int ``N``: exactly ``N`` bins (``min_n_bins = max_n_bins = N``);
    - a mapping: ``{min_bin: X, max_bin: Y}`` for an explicit range.

    A feature without an entry returns ``None`` so callers can skip it.
    """
    spec = FEATURE_BINS.get(feature)
    if spec is None:
        return None
    if isinstance(spec, dict):
        return int(spec.get("min_bin", MIN_BIN)), int(spec.get("max_bin", MAX_BIN))
    n_bins = int(spec)
    return n_bins, n_bins
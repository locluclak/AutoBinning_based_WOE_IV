# run with conda env name 'sacom'
import json
from pathlib import Path

from core.config_loader import INPUT_FILE, OUTPUT_JSON, label_name
from core.io_utils import load_data
from woe_all.select_best import build_best_config


def main():
    df = load_data(INPUT_FILE)
    payload, skipped = build_best_config(df, label_name)

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"JSON config: {OUTPUT_JSON} ({len(payload['features'])} features)")
    if skipped:
        print(f"Skipped {len(skipped)} feature(s):")
        for feature, reason in skipped:
            print(f"  {feature}: {reason}")


if __name__ == "__main__":
    main()

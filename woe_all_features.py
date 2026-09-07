# run with conda env name 'sacom'
from pathlib import Path

from core.config_loader import INPUT_FILE, OUTPUT_FILE, label_name
from core.io_utils import load_data
from woe_all.report import build_report


def main():
    df = load_data(INPUT_FILE)
    html = build_report(df, label_name)
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"HTML report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
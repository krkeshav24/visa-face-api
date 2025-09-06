#!/usr/bin/env python3
import os
import json
import csv

# Path to your saved Images folder
IMAGES_DIR = "/opt/visa-face-api/Images"
OUTPUT_CSV = "/opt/visa-face-api/debug_report.csv"

def main():
    rows = []
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(IMAGES_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            debug = data.get("debug", {})
            details = debug.get("classify_details", {})

            row = {
                "json_file": fname,
                "image_file": fname.replace(".json", ".jpg"),
                "face_shape": data.get("face_shape"),
                "confidence": debug.get("confidence"),
                "aspect": details.get("aspect"),
                "lf": details.get("lf"),
                "fj": details.get("fj"),
                "theta": details.get("theta"),
                "jr": details.get("jr"),
                "percentages": details.get("percentages"),
            }
            rows.append(row)
        except Exception as e:
            print(f"Failed to parse {fname}: {e}")

    if not rows:
        print("No JSON files found in Images directory.")
        return

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "json_file", "image_file", "face_shape", "confidence",
            "aspect", "lf", "fj", "theta", "jr", "percentages"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} records to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

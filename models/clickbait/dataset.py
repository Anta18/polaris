from datasets import load_dataset
import pandas as pd

def convert_marksverdhei_to_csv(output_csv="clickbait_mverdhei.csv"):
    print("Loading dataset: marksverdhei/clickbait_title_classification")

    ds = load_dataset("marksverdhei/clickbait_title_classification")
    dataset = ds["train"]

    rows = []

    for item in dataset:
        title = item["title"]
        label = int(item["clickbait"])  # 1 for clickbait, 0 for non-clickbait

        rows.append({
            "title": title.strip(),
            "content": "",  # No separate content field in this dataset
            "label": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Saved: {output_csv}")
    print("Samples:", len(df))
    print("Clickbait ratio:", df.label.mean())

convert_marksverdhei_to_csv()

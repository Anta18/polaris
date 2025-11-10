"""Utility script to prepare and format data for training."""
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def create_sample_dataset(output_dir="data"):
    """Create a sample dataset structure for reference."""
    os.makedirs(output_dir, exist_ok=True)
    
  
    sample_data = {
        "title": [
            # Clickbait examples
            "You Won't Believe What Happened Next!",
            "This One Simple Trick Will Change Your Life Forever!",
            "SHOCKING: The Truth They Don't Want You to Know!",
            "10 Things Doctors Don't Want You to Know",
            "This Will Blow Your Mind!",
            "What Happened Next Will Shock You",
            "The Reason Why Will Surprise You",
            "Doctors Hate This One Trick",
            
            # Clean examples
            "Scientists Discover New Breakthrough in Medicine",
            "Local News: City Council Meeting Scheduled",
            "Breaking: Major Policy Announcement Today",
            "Economy Shows Strong Growth in Q4",
            "New Study Reveals Benefits of Exercise",
            "City Announces New Park Opening",
            "Weather Forecast for the Weekend",
            "Local Business Opens Downtown Location"
        ],
        "content": [
            # Corresponding content (simplified)
            "Clickbait article content...",
            "Clickbait trick article content...",
            "Sensational article content...",
            "List-based clickbait content...",
            "Mind-blowing clickbait content...",
            "Shocking clickbait content...",
            "Surprising clickbait content...",
            "Trick-based clickbait content...",
            
            "Scientific article content with details...",
            "Local news content with meeting details...",
            "Policy announcement with full details...",
            "Economic report with statistics...",
            "Study details and findings...",
            "Park opening announcement with details...",
            "Detailed weather forecast...",
            "Business opening announcement..."
        ],
        "label": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
    
    # Save to CSV
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"✓ Sample dataset created in {output_dir}/")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - val.csv: {len(val_df)} samples")
    print(f"  - test.csv: {len(test_df)} samples")
    print("\n⚠️  This is a sample dataset. Replace with your actual clickbait detection dataset!")

def convert_json_to_csv(json_file, output_csv):
    """Convert JSON dataset to CSV format."""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✓ Converted {json_file} to {output_csv}")

def format_dataset(input_file, output_file, title_col="title", content_col="content", label_col="label"):
    """Format dataset to required structure."""
    df = pd.read_csv(input_file)
    
    
    if title_col not in df.columns or content_col not in df.columns or label_col not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Required columns not found. Please specify correct column names.")
    
    
    formatted_df = pd.DataFrame({
        "title": df[title_col],
        "content": df[content_col],
        "label": df[label_col]
    })
    
    
    formatted_df["label"] = formatted_df["label"].apply(lambda x: 1 if x in [1, "clickbait", "Clickbait", True] else 0)
    
    formatted_df.to_csv(output_file, index=False)
    print(f"✓ Formatted dataset saved to {output_file}")
    print(f"  - Total samples: {len(formatted_df)}")
    print(f"  - Clickbait (1): {formatted_df['label'].sum()}")
    print(f"  - Clean (0): {len(formatted_df) - formatted_df['label'].sum()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    parser.add_argument("--format", type=str, help="Format existing dataset file")
    parser.add_argument("--output", type=str, default="data/formatted.csv", help="Output file")
    parser.add_argument("--title_col", type=str, default="title", help="Title column name")
    parser.add_argument("--content_col", type=str, default="content", help="Content column name")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
    elif args.format:
        format_dataset(args.format, args.output, args.title_col, args.content_col, args.label_col)
    else:
        print("Use --create_sample to create a sample dataset or --format to format an existing dataset")


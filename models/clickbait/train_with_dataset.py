"""Quick training script for clickbait_mverdhei.csv dataset."""
import subprocess
import sys
import os

def main():
    dataset_file = "clickbait_mverdhei.csv"
    
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file '{dataset_file}' not found!")
        print("Please ensure the dataset file is in the current directory.")
        sys.exit(1)
    
    print("=" * 60)
    print("Training LoRA Adapter for Clickbait Detection")
    print("=" * 60)
    print(f"Dataset: {dataset_file}")
    print("=" * 60)
    print()
    
    # Training command
    cmd = [
        sys.executable,
        "train_lora.py",
        "--dataset_file", dataset_file,
        "--epochs", "5",
        "--batch_size", "16",
        "--learning_rate", "2e-4",
        "--split_ratio", "0.2"
    ]
    
    print("Running training command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        print(f"Adapter saved to: models/lora_adapter/")
        print()
        print("Next steps:")
        print("  1. Test the API: python run.py")
        print("  2. Visit: http://localhost:8000/docs")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ Training failed!")
        print("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()


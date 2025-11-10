"""Script to download and set up the base model and LoRA adapter."""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
import shutil

# Configuration
BASE_MODEL_NAME = "distilbert-base-uncased"  
BASE_MODEL_PATH = "models/base_model"
LORA_ADAPTER_PATH = "models/lora_adapter"

def download_base_model(model_name: str, output_path: str):
    """Download base model from HuggingFace."""
    print(f"Downloading base model: {model_name}")
    print(f"Output path: {output_path}")
    
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
        
        # Download model
        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
        model.save_pretrained(output_path)
        
        print(f"✓ Base model downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading base model: {e}")
        return False

def setup_lora_adapter(output_path: str):
    """Create empty LoRA adapter directory structure."""
    os.makedirs(output_path, exist_ok=True)
    print(f"✓ Created LoRA adapter directory: {output_path}")
    print("⚠ NOTE: You need to train or download the LoRA adapter separately.")
    print("   Place the LoRA adapter files (adapter_config.json, adapter_model.bin) in this directory.")

def main():
    print("=" * 60)
    print("Model Setup Script")
    print("=" * 60)
    
    # Download base model
    success = download_base_model(BASE_MODEL_NAME, BASE_MODEL_PATH)
    
    if success:
        print("\n" + "=" * 60)
        print("Base model setup complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Base model setup failed. Please check the error above.")
        print("=" * 60)
        return
    
    
    print("\nSetting up LoRA adapter directory...")
    setup_lora_adapter(LORA_ADAPTER_PATH)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. If you have a trained LoRA adapter, copy the files to:")
    print(f"   {LORA_ADAPTER_PATH}/")
    print("2. Required LoRA adapter files:")
    print("   - adapter_config.json")
    print("   - adapter_model.bin (or adapter_model.safetensors)")
    print("3. Once models are in place, you can run the API:")
    print("   python run.py")
    print("=" * 60)

if __name__ == "__main__":
    main()


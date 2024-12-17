import os

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint directory structure and print relevant information"""
    print(f"\nVerifying checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Path does not exist: {checkpoint_path}")
        return False
        
    print("Directory contents:")
    for item in os.listdir(checkpoint_path):
        full_path = os.path.join(checkpoint_path, item)
        if os.path.isdir(full_path):
            print(f"  📁 {item}/")
            for subitem in os.listdir(full_path):
                print(f"    - {subitem}")
        else:
            print(f"  📄 {item}")
            
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_path, f))]
    
    if missing_files:
        print(f"\nWARNING: Missing required files: {', '.join(missing_files)}")
        return False
    
    print("\nCheckpoint verification successful!")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        verify_checkpoint(sys.argv[1])
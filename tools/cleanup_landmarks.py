import os
from pathlib import Path

def cleanup_landmarks():
    """Remove landmark files that don't have corresponding images."""
    base_dir = Path("data_output")
    removed_count = 0
    
    # Iterate through all person directories
    for person_dir in base_dir.glob("person*"):
        # Iterate through handedness (left/right)
        for hand_dir in person_dir.glob("*"):
            # Iterate through all number directories
            for number_dir in hand_dir.glob("[0-5]"):
                landmark_dir = number_dir / "landmarks"
                image_dir = number_dir / "images"
                
                # Get all landmark files
                for landmark_file in landmark_dir.glob("*.json"):
                    # Construct corresponding image filename
                    image_file = image_dir / f"{landmark_file.stem}.jpg"
                    
                    # If image doesn't exist, remove the landmark file
                    if not image_file.exists():
                        print(f"Removing {landmark_file} (no corresponding image)")
                        landmark_file.unlink()
                        removed_count += 1
    
    print(f"\nCleanup complete. Removed {removed_count} landmark files without corresponding images.")

if __name__ == "__main__":
    cleanup_landmarks() 
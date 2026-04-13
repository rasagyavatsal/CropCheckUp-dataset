import os
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import io
import csv
import sys

def process_image(input_path, output_path, session):
    """
    Segments the foreground leaf, crops it to its bounding box, 
    and resizes/pads it to 256x256 with a black background.
    """
    try:
        # Load image
        with open(input_path, 'rb') as i:
            input_data = i.read()
        
        # Remove background using U2Net (via rembg)
        # Session is initialized with "u2net" model
        output_data = remove(input_data, session=session)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(output_data))
        
        # Ensure we have RGBA to work with alpha
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # Get alpha channel as numpy array for metrics
        alpha = np.array(img)[:, :, 3]
        foreground_pixels = np.count_nonzero(alpha)
        total_pixels = alpha.size
        foreground_ratio = foreground_pixels / total_pixels
        
        # 1. CROP: Get bounding box of the leaf (non-transparent part)
        bbox = img.getbbox() # (left, top, right, bottom)
        if bbox:
            img = img.crop(bbox)
        
        # 2. RESIZE & PAD: Create a black background 256x256 image (RGB)
        # This matches the PlantVillage standard
        final_img = Image.new("RGB", (256, 256), (0, 0, 0))
        
        # Resize leaf to fit in 256x256 while maintaining aspect ratio
        # Use 240 as target to leave a small margin around the leaf
        img.thumbnail((240, 240), Image.Resampling.LANCZOS)
        
        # Center the leaf
        offset = ((256 - img.width) // 2, (256 - img.height) // 2)
        
        # Paste leaf onto black background (masking with alpha)
        final_img.paste(img.convert("RGB"), offset, img)
        
        # Save output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_img.save(output_path, "JPEG", quality=95)
        
        return {
            'filename': os.path.basename(input_path),
            'class': os.path.basename(os.path.dirname(input_path)),
            'foreground_pixels': foreground_pixels,
            'foreground_ratio': f"{foreground_ratio:.4f}",
            'status': 'success'
        }
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return {
            'filename': os.path.basename(input_path),
            'class': os.path.basename(os.path.dirname(input_path)),
            'status': f'error: {str(e)}'
        }

def main():
    input_root = 'mango'
    output_root = 'mango_segmented'
    csv_path = 'segmentation_foreground_area_check.csv'
    
    # Initialize rembg session with u2net
    print("Initializing U2Net session (this may download the model on first run)...")
    session = new_session("u2net")
    
    # Collect all image paths
    image_tasks = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                
                # Following PlantVillage naming convention for output
                rel_path = os.path.relpath(root, input_root)
                base_name = os.path.splitext(file)[0]
                new_filename = f"{base_name}_final_masked.jpg"
                output_path = os.path.join(output_root, rel_path, new_filename)
                
                image_tasks.append((input_path, output_path))
    
    total = len(image_tasks)
    print(f"Found {total} images to process.")
    
    results = []
    
    # Process images one by one
    for i, (in_p, out_p) in enumerate(image_tasks):
        sys.stdout.write(f"\r[{i+1}/{total}] Processing {in_p}...")
        sys.stdout.flush()
        
        res = process_image(in_p, out_p, session)
        results.append(res)
        
        # Periodically save CSV to avoid data loss
        if (i + 1) % 10 == 0 or (i + 1) == total:
            keys = results[0].keys() if results else []
            with open(csv_path, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)

    print(f"\nProcessing complete. Segmented images saved to '{output_root}'.")
    print(f"Report saved to '{csv_path}'.")

if __name__ == "__main__":
    main()

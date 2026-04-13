import os
import io
import numpy as np
from PIL import Image
from rembg import remove, new_session
import tensorflow as tf
from tqdm import tqdm

def process_images():
    input_base_dir = 'mango'
    output_base_dir = 'mango_processed'
    
    # Initialize rembg session with u2net
    print("Initialising U2NET model...")
    session = new_session("u2net")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Get class directories
    classes = [c for c in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, c))]
    
    for class_name in classes:
        input_class_dir = os.path.join(input_base_dir, class_name)
        
        # Format output folder name to match PlantVillage style (Species___Disease)
        # Replacing spaces with underscores if any
        formatted_class_name = class_name.replace(' ', '_')
        output_class_name = f"Mango___{formatted_class_name}"
        output_class_dir = os.path.join(output_base_dir, output_class_name)
        
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            
        print(f"\nProcessing class: {class_name} -> {output_class_name}")
        images = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        corrupted_count = 0
        for img_name in tqdm(images, desc=f"Class: {class_name}"):
            img_path = os.path.join(input_class_dir, img_name)
            # Use .jpg extension as per PlantVillage
            target_name = os.path.splitext(img_name)[0] + '.jpg'
            output_path = os.path.join(output_class_dir, target_name)
            
            try:
                # Load image
                with open(img_path, 'rb') as f:
                    input_data = f.read()
                
                # Remove background using rembg (U2NET)
                output_data = remove(input_data, session=session)
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(output_data)).convert("RGBA")
                
                # Find bounding box of non-transparent areas
                alpha = img.getchannel('A')
                bbox = alpha.getbbox()
                
                if bbox:
                    # Crop to bounding box
                    img = img.crop(bbox)
                
                # Create a square background (black) just like PlantVillage
                max_dim = max(img.size)
                square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
                
                # Center the cropped image
                x_offset = (max_dim - img.size[0]) // 2
                y_offset = (max_dim - img.size[1]) // 2
                
                # Paste the cropped image onto the square background
                # We use the image itself as a mask to preserve transparency
                square_img.paste(img, (x_offset, y_offset), mask=img)
                
                # Resize to 256x256 (standard PlantVillage size)
                square_img = square_img.resize((256, 256), Image.Resampling.LANCZOS)
                
                # Save as JPEG
                square_img.save(output_path, "JPEG", quality=95)
                
                # check if any corruption comes in TensorFlow’s JPEG decoder
                try:
                    raw_img = tf.io.read_file(output_path)
                    tf.io.decode_jpeg(raw_img)
                except Exception as e:
                    print(f"\n[WARNING] TensorFlow JPEG decoder corruption detected for {output_path}: {e}")
                    corrupted_count += 1
                    
            except Exception as e:
                print(f"\n[ERROR] Failed to process {img_path}: {e}")
        
        if corrupted_count > 0:
            print(f"Total TF-corrupted images in {class_name}: {corrupted_count}")

if __name__ == "__main__":
    process_images()

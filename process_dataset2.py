import os
import io
import numpy as np
from PIL import Image
from rembg import remove, new_session
import tensorflow as tf
from tqdm import tqdm

def process_images():
    input_base_dir = 'dataset2'
    output_base_dir = 'dataset2_processed'
    
    # Initialize rembg session with u2net
    print("Initialising U2NET model...")
    session = new_session("u2net")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Mapping for crops based on metadata.csv
    crop_mapping = {
        'Bottle Gourd': 'Bottle_Gourd',
        'papaya': 'Papaya',
        'Zucchini': 'Zucchini'
    }

    # Iterate through crops
    for crop_folder, crop_name in crop_mapping.items():
        crop_path = os.path.join(input_base_dir, crop_folder)
        
        # Handle the nested Zucchini folder structure observed in dataset2
        if crop_folder == 'Zucchini':
            actual_crop_path = os.path.join(crop_path, 'Zucchini')
        else:
            actual_crop_path = crop_path
            
        if not os.path.exists(actual_crop_path):
            print(f"Warning: Path {actual_crop_path} does not exist.")
            continue

        class_folders = [f for f in os.listdir(actual_crop_path) if os.path.isdir(os.path.join(actual_crop_path, f))]
        
        for class_folder in class_folders:
            # Determine the display class name based on metadata.csv labels
            display_class_name = class_folder.replace('_', ' ')
            
            # Normalize display class name to match metadata.csv where possible
            if crop_name == 'Bottle_Gourd':
                if class_folder == 'Angular_Leaf_spot':
                    display_class_name = 'Hole Angular Leaf Spot'
            elif crop_name == 'Zucchini':
                if class_folder == 'Downy_Midew': # Fix typo in folder name
                    display_class_name = 'Downy Mildew'
            elif crop_name == 'Papaya':
                if class_folder == 'healthy_leaf':
                    display_class_name = 'Healthy Leaf'
                elif class_folder == 'pathogen_symptoms':
                    display_class_name = 'Pathogen Symptoms'
            
            # Format output folder name to match PlantVillage style (Species___Disease)
            formatted_class_name = display_class_name.replace(' ', '_')
            output_class_name = f"{crop_name}___{formatted_class_name}"
            output_class_dir = os.path.join(output_base_dir, output_class_name)
            
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                
            input_class_dir = os.path.join(actual_crop_path, class_folder)
            print(f"\nProcessing class: {crop_folder}/{class_folder} -> {output_class_name}")
            
            images = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            corrupted_count = 0
            for img_name in tqdm(images, desc=f"Class: {class_folder}"):
                img_path = os.path.join(input_class_dir, img_name)
                # Use .jpg extension as per PlantVillage
                target_name = os.path.splitext(img_name)[0] + '.jpg'
                output_path = os.path.join(output_class_dir, target_name)
                
                try:
                    # Load image
                    with open(img_path, 'rb') as f:
                        input_data = f.read()
                    
                    # Remove background using rembg (U2NET)
                    # Even if the image is already isolated (png), this handles cropping and normalization
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
                print(f"Total TF-corrupted images in {output_class_name}: {corrupted_count}")

if __name__ == "__main__":
    process_images()

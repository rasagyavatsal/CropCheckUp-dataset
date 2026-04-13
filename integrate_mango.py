import os
import shutil

def integrate():
    src_root = 'mango_segmented'
    dst_root = 'plantvillage'
    
    if not os.path.exists(src_root):
        print(f"Error: {src_root} does not exist.")
        return

    # Map for special cases
    class_mapping = {
        'Healthy': 'healthy',
        'Bacterial Canker': 'Bacterial_Canker',
        'Die Back': 'Die_Back',
        'Gall Midge': 'Gall_Midge',
        'Powdery Mildew': 'Powdery_Mildew',
        'Sooty Mould': 'Sooty_Mould',
        'Anthracnose': 'Anthracnose'
    }

    subdirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    
    for subdir in subdirs:
        # Construct consistent name
        condition = class_mapping.get(subdir, subdir.replace(' ', '_'))
        new_name = f"Mango___{condition}"
        src_path = os.path.join(src_root, subdir)
        dst_path = os.path.join(dst_root, new_name)
        
        print(f"Moving {src_path} -> {dst_path}")
        if os.path.exists(dst_path):
            print(f"Warning: {dst_path} already exists. Merging contents.")
            for item in os.listdir(src_path):
                s = os.path.join(src_path, item)
                d = os.path.join(dst_path, item)
                shutil.move(s, d)
            os.rmdir(src_path)
        else:
            shutil.move(src_path, dst_path)

    print("Integration complete.")

if __name__ == "__main__":
    integrate()

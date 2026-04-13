import os
from PIL import Image
from collections import Counter
import json

def analyze_images(root_dir):
    dimensions = []
    formats = Counter()
    total_images = 0
    
    print(f"Analyzing images in: {root_dir}")
    
    for subdir, dirs, files in os.walk(root_dir):
        # Skip top level itself if needed, but here we want all nested
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        w, h = img.size
                        dimensions.append((w, h))
                        formats[img.format] += 1
                        total_images += 1
                except Exception as e:
                    print(f"Error opening {file_path}: {e}")
                
                if total_images % 1000 == 0 and total_images > 0:
                    print(f"Processed {total_images} images...")

    if not dimensions:
        print("No images found.")
        return

    # Statistics
    widths = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    
    unique_sizes = Counter(dimensions)
    
    stats = {
        "total_images": total_images,
        "width": {
            "min": min(widths),
            "max": max(widths),
            "avg": sum(widths) / len(widths)
        },
        "height": {
            "min": min(heights),
            "max": max(heights),
            "avg": sum(heights) / len(heights)
        },
        "formats": dict(formats),
        "top_10_sizes": unique_sizes.most_common(10)
    }
    
    print("\n--- RESULTS ---")
    print(json.dumps(stats, indent=4))
    
    # Save detailed sizes if needed, but top 10 is usually enough for overview
    if len(unique_sizes) > 1:
        print(f"\nFound {len(unique_sizes)} unique dimension combinations.")
    else:
        print(f"\nAll images have the same dimensions: {dimensions[0]}")

if __name__ == "__main__":
    analyze_images("/Users/rasagyavatsal/CropCheckUp-dataset/plantvillage")

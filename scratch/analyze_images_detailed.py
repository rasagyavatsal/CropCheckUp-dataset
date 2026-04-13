import os
from PIL import Image
from collections import Counter
import json

def analyze_directory(root_dir):
    all_stats = {}
    
    print(f"Analyzing directories in: {root_dir}")
    
    # Analyze direct subdirectories first
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subdir_name in sorted(subdirs):
        subdir_path = os.path.join(root_dir, subdir_name)
        dimensions = []
        formats = Counter()
        count = 0
        
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            dimensions.append(img.size)
                            formats[img.format] += 1
                            count += 1
                    except:
                        pass
        
        if count > 0:
            widths = [d[0] for d in dimensions]
            heights = [d[1] for d in dimensions]
            unique_sizes = Counter(dimensions)
            
            all_stats[subdir_name] = {
                "count": count,
                "dims": {
                    "common": unique_sizes.most_common(3),
                    "min_w": min(widths),
                    "max_w": max(widths),
                    "min_h": min(heights),
                    "max_h": max(heights)
                },
                "formats": dict(formats)
            }
            print(f"Finished {subdir_name}: {count} images")

    return all_stats

if __name__ == "__main__":
    base_path = "/Users/rasagyavatsal/CropCheckUp-dataset"
    results = {}
    
    for folder in ["plantvillage", "dataset2", "dataset2_processed"]:
        path = os.path.join(base_path, folder)
        if os.path.exists(path):
            results[folder] = analyze_directory(path)
            
    with open(os.path.join(base_path, "scratch/image_analysis.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nAnalysis complete. Results saved to scratch/image_analysis.json")

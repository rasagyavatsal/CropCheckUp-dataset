==========================================================================
DATASET NAME: 
Combined Bottle-Gourd, Zucchini, and Papaya Leaf Disease Dataset
==========================================================================

AUTHORS:
1. Md Masum Billah (Corresponding Author) - billah.masumcu@gmail.com
2. Md. Anisur Rahman
3. Saifuddin Sagor
4. Mohammad Shorif Uddin

AFFILIATIONS:
- Daffodil International University, Dhaka, Bangladesh
- Jahangirnagar University, Savar, Dhaka, Bangladesh

==========================================================================
1. DATASET DESCRIPTION
==========================================================================
This dataset contains high-resolution images of healthy and diseased leaves for three major crops: Bottle-Gourd (Lagenaria siceraria), Zucchini (Cucurbita pepo), and Papaya (Carica papaya). 

The dataset was created to facilitate research in AI-driven agricultural disease detection, computer vision, and machine learning.

- Total Original Images: 2,144
- Total Augmented Images: 23,000 (Expanded to ~1,000 images per class)
- Number of Classes: 23
- Location: Daffodil International University Smart City (GPS: 23.8769° N, 90.3113° E) and agricultural fields in Bangladesh.
- Capture Device: SONY ALPHA 7 II (Full-frame camera).
- Validation: Verified by 3 senior agronomists using majority voting consensus.

==========================================================================
2. FOLDER STRUCTURE
==========================================================================
The dataset is organized into two main directories: "Original_Images" and "Augmented_Images". Inside each, images are sorted by Crop Name and then by Disease Class.

├── Original_Images/
│   ├── Bottle_Gourd/
│   │   ├── Alternaria Leaf Blight/
│   │   ├── Hole Angular Leaf Spot/
│   │   ├── Anthracnose/
│   │   ├── Downy Mildew/
│   │   ├── Early Alternaria Leaf Blight/
│   │   ├── Fungal Damage Leaf/
│   │   ├── Healthy/
│   │   └── Mosaic Virus/
│   ├── Zucchini/
│   │   ├── Angular Leaf Spot/
│   │   ├── Anthracnose/
│   │   ├── Downy Mildew/
│   │   ├── Dry Leaf/
│   │   ├── Healthy/
│   │   ├── Insect Damage/
│   │   ├── Iron Chlorosis Damage/
│   │   ├── Xanthomonas Leaf Spot/
│   │   └── Yellow Mosaic Virus/
│   └── Papaya/
│       ├── Bacterial Blight/
│       ├── Carica Insect Hole/
│       ├── Curled Yellow Spot/
│       ├── Healthy Leaf/
│       ├── Pathogen Symptoms/
│       └── Yellow Necrotic Spots/
│
├── Augmented_Images/
│   ├── [Same structure as Original_Images, containing 1,000 images per class]
│
└── metadata.csv  (Contains image-level details: GPS, Lighting, Timestamp, Camera settings)

==========================================================================
3. METHODOLOGY & AUGMENTATION DETAILS
==========================================================================
> Image Acquisition: 
Captured under natural lighting conditions (Morning: 08:00–11:00 AM & Afternoon: 03:00–05:00 PM) to capture realistic variability.

> Pre-processing & Augmentation:
Software used: Python (v3.9) with OpenCV and PIL (Pillow) libraries.
All images were resized to 512x512 pixels. The following augmentation parameters were applied to address class imbalance:

- Rotation: Random rotation of ±15° and ±30°.
- Zoom/Scaling: Random scaling factors of 1.1x and 1.3x.
- Flipping: Horizontal and Vertical flipping.
- Brightness/Color: Random adjustment factors between 0.7 and 1.3.
- Shifting: Translation along x and y axes by ±10 pixels.
- Cropping: Center cropping at ratios of 0.6 and 0.8.

==========================================================================
4. FILE USAGE INSTRUCTIONS
==========================================================================
- For Training: Use the 'Augmented_Images' folder for training deep learning models (CNNs, ViTs, etc.) to ensure class balance.
- For Testing/Validation: It is recommended to use a split from the 'Original_Images' folder to test model performance on real-world non-augmented data.
- Metadata: Refer to 'metadata.csv' to filter images based on lighting conditions or capture time.

==========================================================================
5. LICENSE & CITATION
==========================================================================
This dataset is openly available on Mendeley Data. If you use this dataset in your research, please cite the associated Data in Brief article.

(C) 2024-2025 The Authors.
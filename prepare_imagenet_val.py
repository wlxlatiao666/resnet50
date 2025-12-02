#!/usr/bin/env python3
"""
Reorganize ImageNet validation set into class subdirectories.

Usage:
    python prepare_imagenet_val.py <imagenet_root>

Example:
    python prepare_imagenet_val.py /path/to/imagenet

This script expects:
- <imagenet_root>/val/ containing validation images (ILSVRC2012_val_*.JPEG)
- Ground truth file at one of these locations:
  1. <imagenet_root>/ILSVRC2012_validation_ground_truth.txt
  2. <imagenet_root>/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
  3. Will download if not found
"""

import os
import sys
import shutil
import urllib.request


def download_ground_truth(output_path):
    """Download validation ground truth file"""
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
    print(f"Downloading ground truth labels from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def find_ground_truth_file(imagenet_root):
    """Find or download the ground truth file"""
    possible_paths = [
        os.path.join(imagenet_root, 'ILSVRC2012_validation_ground_truth.txt'),
        os.path.join(imagenet_root, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt'),
        os.path.join(imagenet_root, 'imagenet_2012_validation_synset_labels.txt'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found ground truth file: {path}")
            return path
    
    # Try to download
    download_path = os.path.join(imagenet_root, 'imagenet_2012_validation_synset_labels.txt')
    if download_ground_truth(download_path):
        return download_path
    
    return None


def load_synset_mapping():
    """Load the mapping from class index to synset ID (WordNet ID)"""
    # This is the standard ImageNet 1000-class mapping
    # Synsets in order of class index (0-999)
    synsets_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
    
    # For simplicity, we'll use a hardcoded approach
    # The ground truth file from TensorFlow already contains synset IDs directly
    return None


def reorganize_val_folder(val_dir, ground_truth_file):
    """Reorganize validation images into class subdirectories"""
    
    # Read ground truth labels (synset IDs, one per line)
    with open(ground_truth_file, 'r') as f:
        synsets = [line.strip() for line in f]
    
    print(f"Loaded {len(synsets)} ground truth labels")
    
    # Get all validation images (sorted to match ground truth order)
    val_images = sorted([f for f in os.listdir(val_dir) 
                        if f.startswith('ILSVRC2012_val_') and f.endswith('.JPEG')])
    
    if len(val_images) == 0:
        print(f"Error: No validation images found in {val_dir}")
        print("Expected files like: ILSVRC2012_val_00000001.JPEG")
        return False
    
    print(f"Found {len(val_images)} validation images")
    
    if len(val_images) != len(synsets):
        print(f"Warning: Number of images ({len(val_images)}) doesn't match labels ({len(synsets)})")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create class directories and move images
    moved_count = 0
    for i, img_name in enumerate(val_images):
        if i >= len(synsets):
            print(f"Warning: No label for image {img_name}, skipping")
            continue
            
        synset = synsets[i]
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(val_dir, synset)
        os.makedirs(class_dir, exist_ok=True)
        
        # Move image to class directory
        src = os.path.join(val_dir, img_name)
        dst = os.path.join(class_dir, img_name)
        
        try:
            shutil.move(src, dst)
            moved_count += 1
            
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(val_images)} images")
        except Exception as e:
            print(f"Error moving {img_name}: {e}")
    
    print(f"\nSuccessfully reorganized {moved_count} images into {len(set(synsets))} class folders")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_imagenet_val.py <imagenet_root>")
        print("\nExample: python prepare_imagenet_val.py /path/to/imagenet")
        print("\nThis will reorganize <imagenet_root>/val/ into class subdirectories")
        sys.exit(1)
    
    imagenet_root = sys.argv[1]
    val_dir = os.path.join(imagenet_root, 'val')
    
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found: {val_dir}")
        sys.exit(1)
    
    # Check if already organized
    subdirs = [d for d in os.listdir(val_dir) 
              if os.path.isdir(os.path.join(val_dir, d)) and not d.startswith('.')]
    
    if len(subdirs) >= 1000:
        print(f"Validation set appears to already be organized ({len(subdirs)} subdirectories found)")
        response = input("Reorganize anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Find ground truth file
    ground_truth_file = find_ground_truth_file(imagenet_root)
    
    if not ground_truth_file:
        print("\nError: Could not find or download ground truth file.")
        print("\nPlease download one of these files:")
        print("1. ILSVRC2012_validation_ground_truth.txt from the ImageNet devkit")
        print("2. Or download from: https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt")
        print(f"\nPlace it in: {imagenet_root}/")
        sys.exit(1)
    
    # Reorganize
    print(f"\nReorganizing validation set in: {val_dir}")
    success = reorganize_val_folder(val_dir, ground_truth_file)
    
    if success:
        print("\n✓ Validation set successfully reorganized!")
        print(f"You can now run: python train.py {imagenet_root} --gpu 0")
    else:
        print("\n✗ Failed to reorganize validation set")
        sys.exit(1)


if __name__ == '__main__':
    main()

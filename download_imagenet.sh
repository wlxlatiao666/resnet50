#!/bin/bash

# Create directory structure
mkdir -p data/imagenet
cd data/imagenet

echo "Checking for Kaggle API..."
if command -v kaggle &> /dev/null; then
    echo "Kaggle API found. Attempting to download ImageNet..."
    echo "Note: You need to have accepted the competition rules on Kaggle."
    kaggle competitions download -c imagenet-object-localization-challenge
    
    echo "Unzipping..."
    unzip imagenet-object-localization-challenge.zip
    
    # Organize validation set (ImageNet devkit usually needed, but we can use a helper script)
    # For now, we assume the standard structure or provide a helper python script later.
else
    echo "Kaggle API not found."
    echo "Please download the ImageNet dataset (ILSVRC2012) manually."
    echo "You need 'ILSVRC2012_img_train.tar' and 'ILSVRC2012_img_val.tar'."
    echo "Place them in $(pwd)"
    echo ""
    echo "Alternatively, install kaggle CLI: pip install kaggle"
    echo "And setup your ~/.kaggle/kaggle.json"
fi

# Helper to extract if tar files exist
if [ -f "ILSVRC2012_img_train.tar" ]; then
    echo "Extracting train..."
    mkdir -p train
    tar -xvf ILSVRC2012_img_train.tar -C train
    # Train images are usually in sub-tars
    cd train
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
fi

if [ -f "ILSVRC2012_img_val.tar" ]; then
    echo "Extracting val..."
    mkdir -p val
    tar -xvf ILSVRC2012_img_val.tar -C val
    # Validation images are usually flat, need a script to move them to folders
    # We will provide prepare_imagenet_val.py for this.
fi
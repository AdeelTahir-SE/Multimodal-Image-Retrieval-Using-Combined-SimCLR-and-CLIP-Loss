#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/coco
cd data/coco

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

extract_zip() {
	local zip_file="$1"
	if command -v unzip >/dev/null 2>&1; then
		unzip -o "$zip_file"
	else
		echo "'unzip' not found. Falling back to Python zip extraction for $zip_file"
		python3 -m zipfile -e "$zip_file" .
	fi
}

extract_zip train2017.zip
extract_zip val2017.zip
extract_zip annotations_trainval2017.zip

echo "MS-COCO 2017 train/val images and annotations downloaded to data/coco"
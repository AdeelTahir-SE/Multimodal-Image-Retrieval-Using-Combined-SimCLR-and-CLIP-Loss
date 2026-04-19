#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/coco
cd data/coco

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -o train2017.zip
unzip -o annotations_trainval2017.zip

echo "MS-COCO 2017 train images and annotations downloaded to data/coco"
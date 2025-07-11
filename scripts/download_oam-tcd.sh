#!/bin/bash

TARGET_DIR="${1:-.}"

mkdir -p "$TARGET_DIR"

wget -P "$TARGET_DIR" https://zenodo.org/records/11617167/files/test.tar.gz
wget -P "$TARGET_DIR" https://zenodo.org/records/11617167/files/train.tar.gz
wget -P "$TARGET_DIR" https://zenodo.org/records/11617167/files/masks.tar.gz
wget -P "$TARGET_DIR" https://zenodo.org/records/11617167/files/images.tar.gz

tar -xvf "$TARGET_DIR/test.tar.gz" -C "$TARGET_DIR"
tar -xvf "$TARGET_DIR/train.tar.gz" -C "$TARGET_DIR"
tar -xvf "$TARGET_DIR/masks.tar.gz" -C "$TARGET_DIR"
tar -xvf "$TARGET_DIR/images.tar.gz" -C "$TARGET_DIR"
#!/bin/bash

FILE_NAME="3droad.mat"
URL="https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1"

if [ ! -f "$FILE_NAME" ]; then
  echo "Downloading '3droad' UCI dataset..."
  curl -L "$URL" -o "$FILE_NAME"
else
  echo "'3droad.mat' already exists in the current directory."
fi

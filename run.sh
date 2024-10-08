#!/bin/bash

python3 -m venv test

source test/bin/activate

pip install --upgrade pip

echo "numpy==2.1.1" >> requirements.txt
echo "opencv-python==4.10.0.84" >> requirements.txt
echo "tqdm==4.66.5" >> requirements.txt

pip install -r requirements.txt

pip install --upgrade -r requirements.txt

cd CV_Assignment02

python3 main_backgroundsubtraction.py

echo 'done'

cd ..

rm -rf requirements.txt

deactivate

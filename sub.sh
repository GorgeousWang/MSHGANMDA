#!/bin/sh
#JSUB -q normal
#JSUB -n 2
#JSUB -e ./result/error-wfy.%J
#JSUB -o ./result/output-wfy.%J
#JSUB -J wfy-ep100
source /apps/software/anaconda3/bin/activate py37torch190
python main.py

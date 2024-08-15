#!/bin/bash

# Read the argument into a variable
num_epoch="$1"
dataset_no="$2"

script_path="/home/nguyen/Projects/Evolutionary Algorithm/MOEAD/main.py"
echo "$script_path"
# Loop from 1 to the limit
echo "Run main.py $num_epoch times parallelly with dataset $dataset_no!"
for (( i=0; i<$num_epoch; i++ )); do
  python3 "$script_path" $i $dataset_no &
done


#!/bin/bash

versions=("FS5" "FS6" "FS7" "FS8")


# Submit unique pairs only

for ((i=0; i<${#versions[@]}; i++))
do
    for ((j=i+1; j<${#versions[@]}; j++))
    do
        v1=${versions[$j]}
        v2=${versions[$i]}

        echo "Submitting $v1 vs $v2"

        sbatch whole-brain.sh $v1 $v2

    done
done

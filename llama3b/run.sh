#!/bin/bash


echo "running neuron masking script"
python main.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
python main.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0


# echo "Running das masking script"
# python main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
# python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
#
# echo "Running sae masking neel's script"
# python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
# python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
#
# echo "Running sae masking openai script"
# python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
# python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-0" -lr "0.001" -lid 0

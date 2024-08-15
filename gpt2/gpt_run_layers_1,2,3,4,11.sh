# !/bin/bash


#python main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
#python main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0

#echo "Layer 1 Running neuron masking script"
#python main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#python main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1 
#
#
#echo "Layer 1 Running das masking script"
#python main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#
#
#echo "Layer 1 Running sae masking neel's script"
#python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#
#
#echo "Layer 1 Running sae masking openai script"
#python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-1" -lr "0.001" -lid 1
#
echo "Layer 2 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
python main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0


echo "Layer 2 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0


echo "Layer 2 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0


echo "Layer 2 Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-0" -lr "0.001" -lid 0


echo "Layer 3 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
python main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3 


echo "Layer 3 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3


#echo "Layer 3 Running sae masking neel's script"
#python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
#python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
#
#
#echo "Layer 3 Running sae masking openai script"
#python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
#python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-3" -lr "0.001" -lid 3
#
#echo "Layer 4 Running neuron masking script"
#python main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#python main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#
#
#echo "Layer 4 Running das masking script"
#python main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#
#
#echo "Layer 4 Running sae masking neel's script"
#python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#
#
#echo "Layer 4 Running sae masking openai script"
#python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 4
#
# cho "Layer 11 Running neuron masking script"
# python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 11
# python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-4" -lr "0.001" -lid 11
#
#
# echo "Layer 11 Running das masking script"
# python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
# python main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#
#
# echo "Layer 11 Running sae masking neel's script"
# python main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
# python main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#

# echo "Layer 11  Running sae masking openai script"
# python main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
# python main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11

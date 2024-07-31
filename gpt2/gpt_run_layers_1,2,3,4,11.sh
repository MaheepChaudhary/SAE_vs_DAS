# !/bin/bash



echo "Layer 1 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1
python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1 


echo "Layer 1 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1
python main_train.py -a continent -tla 61 -method "das masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1


echo "Layer 1 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1


echo "Layer 1 Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 1

echo "Layer 2 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2
python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2


echo "Layer 2 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2
python main_train.py -a continent -tla 61 -method "das masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2


echo "Layer 2 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2


echo "Layer 2 Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 2


echo "Layer 3 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3
python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3 


echo "Layer 3 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3
python main_train.py -a continent -tla 61 -method "das masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3


echo "Layer 3 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3


echo "Layer 3 Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 3

eecho "Layer 4 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4
python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4


echo "Layer 4 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4
python main_train.py -a continent -tla 61 -method "das masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4


echo "Layer 4 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4


echo "Layer 4 Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 4

cho "Layer 11 Running neuron masking script"
python main_train.py -a country -tla 61 -method "neuron masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11
python main_train.py -a continent -tla 61 -method "neuron masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11


echo "Layer 11 Running das masking script"
python main_train.py -a country -tla 61 -method "das masking" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11
python main_train.py -a continent -tla 61 -method "das masking" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11


echo "Layer 11 Running sae masking neel's script"
python main_train.py -a country -tla 61 -method "sae masking neel" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11
python main_train.py -a continent -tla 61 -method "sae masking neel" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11


echo "Layer 11  Running sae masking openai script"
python main_train.py -a country -tla 61 -method "sae masking openai" -e 10 -t train -idd "country" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11
python main_train.py -a continent -tla 61 -method "sae masking openai" -e 10 -t train -idd "continent" -bs 128 -wb "True" -n "t(10-0.1)_lr0.01_layer-0" -lr "0.01" -lid 11



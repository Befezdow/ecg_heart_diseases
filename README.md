# ECG heart diseases prediction

This is a scientific work related to the analysis of ECG data and the prediction of heart diseases using machine learning algorithms.

## Main dependencies
TODO

## Installation
TODO

## Dataset
TODO

Входной датасет: gender (пол), age (возраст), class1, class2, class3, t1 - t60000 (значение графиков отведений ЭКГ, t1-t5000 - первое отведение, t5001-t10000 - второе отведение и т.д.), class1, class2, class3 - сердечные заболевания (три колонки, так как возможна комбинация заболеваний). 
В датасете представлено 12 отведений по 5000 значений на каждое. 
В данный момент осуществляется классификация только по class1 (10 классов), т.е. предсказывается основной диагноз.

## How to train
TODO

## How to test
TODO

## Models
TODO (+ links to snaphots)

## TODO list
- integrate config file and argument parser
- make augmentation after splitting (it saves test dataset from wrong transformations)
- save snapshots every k epochs
- clear requirements.py
- describe dataset (size + columns)
- make installation guide
- attach links to best model
- make train guide
- make test guide
- share logs in tensorboard.dev

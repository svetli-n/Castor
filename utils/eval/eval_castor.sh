#!/bin/bash

ROOT=${ROOT:-"/Users/svetlin/workspace/q-and-a"}
UTILS=$ROOT/Castor/utils
CASTOR=$ROOT/Castor
CASTOR_DATA=$ROOT/Castor-data
CASTOR_MODELS=$ROOT/Castor-models
SQUAD=$ROOT/squad

NUM_NEG=10

TOTAL=0
let TOTAL=NUM_NEG+1
let TOTAL=TOTAL*261

echo "Castor: answer length 30 train on WikiQA test on Squad2 with num_neg=$NUM_NEG and total=$TOTAL"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total $TOTAL --num_neg $NUM_NEG \
        --dest $CASTOR_DATA/datasets/WikiQA/test --answer_min_len 30 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.wikiqa.model --dataset wikiqa --holistic-filters 100 \
        --skip-training --device -1 --skip-dev

echo "Castor: answer length 10 train on WikiQA test on Squad2 with num_neg=$NUM_NEG and total=$TOTAL"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total $TOTAL --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/WikiQA/test --answer_min_len 10 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.wikiqa.model --dataset wikiqa --holistic-filters 100 \
        --skip-training  --device -1 --skip-dev


echo "Castor: answer length 30 train on TrecQA test on Squad2 with num_neg=$NUM_NEG and total=$TOTAL"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total $TOTAL --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/TrecQA/raw-test --answer_min_len 30 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.trecqa.model --dataset trecqa  --holistic-filters 200 \
        --skip-training --device -1 --skip-dev

echo "Castor: answer length 10 train on TrecQA test on Squad2 with num_neg=$NUM_NEG and total=$TOTAL"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total $TOTAL --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/TrecQA/raw-test --answer_min_len 10 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.trecqa.model --dataset trecqa  --holistic-filters 200 \
        --skip-training --device -1 --skip-dev

#Expected output

#Castor: answer length 30 train on WikiQA test on Squad2 with num_neg=5 and total=1566
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.7202  0.7202  0.4024707135700044
#Castor: answer length 10 train on WikiQA test on Squad2 with num_neg=5 and total=1566
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.6298  0.6298  0.5799435406047875
#Castor: answer length 30 train on TrecQA test on Squad2 with num_neg=5 and total=1566
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.7821  0.7821  1.2562036627814883
#Castor: answer length 10 train on TrecQA test on Squad2 with num_neg=5 and total=1566
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.6579  0.6579  1.6677128486097392


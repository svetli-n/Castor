#!/bin/bash

ROOT=${ROOT:-"/Users/svetlin/workspace/q-and-a"}
UTILS=$ROOT/Castor/utils
CASTOR=$ROOT/Castor
CASTOR_DATA=$ROOT/Castor-data
CASTOR_MODELS=$ROOT/Castor-models
SQUAD=$ROOT/squad


NUM_NEG=${NUM_NEG:-"10"}
NUM_QUESTIONS=${NUM_QUESTIONS:-"1"}

echo "Castor: answer length 30 train on WikiQA test on Squad2 with num_neg=$NUM_NEG and num_questions=$NUM_QUESTIONS"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --num_questions $NUM_QUESTIONS --num_neg $NUM_NEG \
        --dest $CASTOR_DATA/datasets/WikiQA/test --answer_min_len 30 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.wikiqa.model --dataset wikiqa --holistic-filters 100 \
        --skip-training --device -1 --skip-dev --keep-results

echo "Castor: answer length 10 train on WikiQA test on Squad2 with num_neg=$NUM_NEG and num_questions=$NUM_QUESTIONS"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --num_questions $NUM_QUESTIONS --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/WikiQA/test --answer_min_len 10 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.wikiqa.model --dataset wikiqa --holistic-filters 100 \
        --skip-training  --device -1 --skip-dev --keep-results


echo "Castor: answer length 30 train on TrecQA test on Squad2 with num_neg=$NUM_NEG and num_questions=$NUM_QUESTIONS"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --num_questions $NUM_QUESTIONS --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/TrecQA/raw-test --answer_min_len 30 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.trecqa.model --dataset trecqa  --holistic-filters 200 \
        --skip-training --device -1 --skip-dev --keep-results

echo "Castor: answer length 10 train on TrecQA test on Squad2 with num_neg=$NUM_NEG and num_questions=$NUM_QUESTIONS"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --num_questions $NUM_QUESTIONS --num_neg $NUM_NEG  \
        --dest $CASTOR_DATA/datasets/TrecQA/raw-test --answer_min_len 10 --format castor

cd $CASTOR

python -W ignore -m mp_cnn $CASTOR_MODELS/mp_cnn/mpcnn.trecqa.model --dataset trecqa  --holistic-filters 200 \
        --skip-training --device -1 --skip-dev --keep-results


#Example call

#NUM_NEG=5 NUM_QUESTIONS=10 sh eval_castor.sh


#Expected output

#Castor: answer length 30 train on WikiQA test on Squad2 with num_neg=5 and num_questions=10
#Total:60:Unique:10.0
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.7083  0.7083  0.4039849599202474
#Castor: answer length 10 train on WikiQA test on Squad2 with num_neg=5 and num_questions=10
#Total:60:Unique:10.0
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.545   0.545   0.5498195648193359
#Castor: answer length 30 train on TrecQA test on Squad2 with num_neg=5 and num_questions=10
#Total:60:Unique:10.0
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.77    0.77    1.2415805816650392
#Castor: answer length 10 train on TrecQA test on Squad2 with num_neg=5 and num_questions=10
#Total:60:Unique:10.0
#INFO - Evaluation metrics for test
#INFO -          map     mrr     cross entropy loss
#INFO - test     0.7833  0.7833  1.5396302541097004

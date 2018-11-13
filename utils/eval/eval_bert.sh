#!/bin/bash

ROOT=${ROOT:-"/Users/svetlin/workspace/q-and-a"}
UTILS=$ROOT/Castor/utils
GLUE_DIR=$ROOT/glue_data
BERT_DIR=$ROOT/bert
BERT_BASE_DIR=$ROOT/bert-data/cased_L-12_H-768_A-12
BERT_CHECKPOINS=$ROOT/bert-data/checkpoint
SQUAD=$ROOT/squad
OUTPUT=/tmp/bert_qnli_output

QA_LENGTH=30
echo "Bert: answer length $QA_LENGTH train on QNLI test on Squad2"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total 1570 --num_neg 5 \
        --dest $GLUE_DIR/QNLI/dev.tsv --answer_min_len $QA_LENGTH --format bert_qnli

python $BERT_DIR/run_classifier.py \
  --task_name=qnli \
  --do_train=false \
  --do_eval=false \
  --do_pred=true \
  --data_dir=$GLUE_DIR/QNLI \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_CHECKPOINS/model.ckpt-10165 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT

python $UTILS/metrics.py --dataset $GLUE_DIR/QNLI/dev.tsv --preds $OUTPUT/pred_results.csv


QA_LENGTH=10
echo "Bert: answer length $QA_LENGTH train on QNLI test on Squad2"

python $UTILS/create_SQuAD_dataset.py --src $SQUAD/train-v2.0.json --total 1570 --num_neg 5 \
        --dest $GLUE_DIR/QNLI/dev.tsv --answer_min_len $QA_LENGTH --format bert_qnli

python $BERT_DIR/run_classifier.py \
  --task_name=qnli \
  --do_train=false \
  --do_eval=false \
  --do_pred=true \
  --data_dir=$GLUE_DIR/QNLI \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_CHECKPOINS/model.ckpt-10165 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT

python $UTILS/metrics.py --dataset $GLUE_DIR/QNLI/dev.tsv --preds $OUTPUT/pred_results.csv


# Expected output

# Bert: answer length 30 train on QNLI test on Squad2
# mrr: 0.7857142857142857
# Bert: answer length 10 train on QNLI test on Squad2
# mrr: 0.6934865900383141


# Add TODO train on WikiQA and TrecQA

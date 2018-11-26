import argparse
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import pickle
from pandas.io.json import json_normalize
from tqdm import tqdm

np.random.seed(4241)


def to_labeled_pairs(args: argparse.ArgumentParser, dff: pd.DataFrame) -> None:

    all_df, num_answers = get_pos_neg_df(args, dff)

    all_df.question.to_csv(os.path.join(args.dest, 'a.toks'), index=False, sep='\t')
    all_df.answer.to_csv(os.path.join(args.dest, 'b.toks'), index=False, sep='\t')
    all_df.id.to_csv(os.path.join(args.dest, 'id.txt'), index=False, sep='\t')
    all_df.label.to_csv(os.path.join(args.dest, 'sim.txt'), index=False, sep='\t')

    print(f'Total:{len(all_df)}:Number of answers:{num_answers}')


def to_bert_qnli(args: argparse.ArgumentParser, dff: pd.DataFrame) -> None:

    all_df, num_answers = get_pos_neg_df(args, dff)

    all_df.label = all_df.label.apply(lambda val: 'entailment' if val == 1 else 'not_entailment')
    all_df.drop(['id'], axis=1, inplace=True)
    all_df.reset_index(drop=False, inplace=True)
    all_df.drop(['index'], axis=1, inplace=True)
    all_df.rename(index=str, columns={'answer': 'sentence'}, inplace=True)
    all_df.to_csv(args.dest, sep='\t', index_label='index')

    print(f'Total:{len(all_df)}:Number of answers:{num_answers}')


def get_pos_neg_df(args: argparse.ArgumentParser, pos_df: pd.DataFrame) -> pd.DataFrame:
    pos_neg_df = pd.DataFrame([], columns=pos_df.columns)
    neg_answers_size = min(args.num_neg, len(pos_df)-1)
    num_answers = neg_answers_size + 1

    for i, row in tqdm(pos_df.iterrows(), total=args.num_questions):
        if i >= args.num_questions:
            break

        current_neg_df = pd.concat([pos_df.iloc[0:i], pos_df.iloc[i + 1:len(pos_df)]], ignore_index=True)
        sampled_current_neg_df = current_neg_df.sample(neg_answers_size)
        sampled_current_neg_df.question = row.question
        sampled_current_neg_df.label = 0
        sampled_current_neg_df = sampled_current_neg_df.append(row, ignore_index=True)
        pos_neg_df = pd.concat([pos_neg_df, sampled_current_neg_df], ignore_index=True)

    return pos_neg_df, num_answers


def to_pickled_dict(args: argparse.ArgumentParser, dff: pd.DataFrame) -> None:
    out_dict = defaultdict(list)
    for i, row in dff.iterrows():
        num_neg = args.num_neg
        question = row['question']
        pos_answer = row['answer']
        out_dict[question].append([pos_answer, 1])
        while num_neg > 0:
            rand_i = np.random.randint(0, len(dff) - 1)
            rand_neg_answer = dff.iloc[rand_i]['answer']
            if rand_neg_answer != pos_answer:
                out_dict[question].append([rand_neg_answer, 0])
                num_neg -= 1
    result_dict = {'test': out_dict}

    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
    pickle.dump(result_dict, open(os.path.join(args.dest, 'test.pkl'), 'wb'))


def positive_df(args: argparse.ArgumentParser) -> pd.DataFrame:
    raw = pd.read_json(args.src)
    dff = json_normalize(raw.data, record_path=['paragraphs', 'qas'])[['question', 'plausible_answers']]
    dff.dropna(inplace=True)
    dff['answer'] = dff.plausible_answers.apply(lambda x: x[0]['text'] + ' .').str.lower().str.replace(',', ' ,')
    dff.drop('plausible_answers', axis=1, inplace=True)
    dff.question = dff.question.str.replace('\?$', '').str.lower()
    dff['label'] = 1
    dff = dff[dff.answer.apply(lambda a: len(a.split()) >= args.answer_min_len)]
    dff.reset_index(drop=True, inplace=True)
    dff.insert(0, 'id', dff.index)
    return dff


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Squad dataset')
    parser.add_argument('--src', type=str, help='Source folder for squad')
    parser.add_argument('--dest', type=str, help='Destination folder')
    parser.add_argument('--num_neg', type=int, help='Number of negative answers per 1 positive')
    parser.add_argument('--num_questions', type=int, help='Number of different questions')
    parser.add_argument('--answer_min_len', type=int, help='Minimum number of words in the answer')
    parser.add_argument('--format', type=str, help='The dataset format, e.g. castor')
    return parser.parse_args()


if __name__ == '__main__':
    '''
    
    Example usage:
    
    python create_SQuAD_dataset.py --src /tmp/train-v2.0.json --dest /tmp/hyperqa --format hyperqa --total 1570 --num_neg 5 --answer_min_len 4
    
    '''

    args = get_args()

    pos_df = positive_df(args)

    if args.format == 'castor':
        to_labeled_pairs(args, pos_df)
    elif args.format == 'bert_qnli':
        to_bert_qnli(args, pos_df)
    elif args.format == 'hyperqa':
        to_pickled_dict(args, pos_df)
    else:
        raise AttributeError('Specify output format')

import argparse
import pandas as pd


def mrr(args: argparse.ArgumentParser) -> float:
    dataset = pd.read_table(args.dataset, quotechar='~')
    preds = pd.read_csv(args.preds, sep=',')
    merged = pd.concat([dataset, preds], axis=1)

    df = merged.groupby('question')['actual', 'prob_1'].apply(
        lambda x: x.sort_values('prob_1', ascending=False)).reset_index()
    mr = df.groupby('question')['actual'].apply(lambda x: 1 / (1 + x.tolist().index(1)))
    mrr = mr.values.mean()
    return mrr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset file')
    parser.add_argument('--preds', type=str, help='Predictions file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f'mrr: {mrr(args)}')

import json
from os.path import join

from tqdm import tqdm

from test_model import build_reranked_data, S1
from utils.common_utils import get_number_of_total_chunks, parse_config


def load_predictions(input_dir, dataset, output_dir):
    raw_data = []
    scores = []

    for i in tqdm(range(get_number_of_total_chunks('gh', dataset))):
        raw_data_scores_i = json.load(
            open(join(output_dir, f'output_{i}.json')))
        raw_data_i = json.load(
            open(join(input_dir, dataset, '3_gengraph', f'rxns_{i}_test.pt.json')))
        for j in range(len(raw_data_i)):
            key = str(j)
            if key in raw_data_scores_i:
                raw_data.append(raw_data_i[j])
                scores.append(raw_data_scores_i[key])
    return raw_data, scores

def main():
    args = parse_config()
    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset = args.dataset
    raw_data, scores = load_predictions(input_dir, dataset, output_dir)

    evaluating(raw_data, dataset, reranking=False)
    reranked = build_reranked_data(raw_data, scores)
    evaluating(reranked, dataset, reranking=True)


if __name__ == '__main__':
    main()

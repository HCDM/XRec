import os
import argparse
import torch

from main import init, load_dataset
from src.decoder import SearchDecoder, ExtractDecoder
from src.dataset import ExtBuilder, basic_builder
from src.evaluate import test_ext_perplexity, test_ext_mrr, test_bleu, test_ext_bleu_upper_bound, test_exp_perplexity, test_ext_var, test_length, test_diversity, test_self_bleu, test_rate_rmse
import config

DIR_PATH = os.path.dirname(__file__)


def get_builder(model_type):
    if model_type in ['CompExp']:
        return ExtBuilder(n_item_exps=30, n_ref_exps=10, n_pos_exps=0)
    else:
        return basic_builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='model name to save/load checkpoints')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('-s', '--search', default='greedy', choices=['greedy', 'sample'], help='decoding search method, only work for BLEU')
    parser.add_argument('-k', '--topk', default=0, type=int)

    parser.add_argument('evals', nargs='+')
    args = parser.parse_args()

    model, misc = init(args.model, args.checkpoint)
    model.eval()

    test_dataset = load_dataset('test')

    model_type = misc['model_config'].MODEL_TYPE
    has_extractor = model_type in ['CompExp']

    # Eval metrics
    for ev in args.evals:
        if ev == 'ext_ppl':
            ppl = test_ext_perplexity(test_dataset, model)
            print('Extraction Perplexity: ', ppl)

        elif ev == 'ext_mrr':
            if has_extractor:
                model = model.extractor

            mrr, rank_dist = test_ext_mrr(test_dataset, model)
            print('Extraction MRR: ', mrr)
            print('Ranking distribution:', rank_dist)

        elif ev == 'ext_var':
            if has_extractor:
                model = model.extractor

            var = test_ext_var(test_dataset, model)
            print('Extraction Variance: ', var)

        elif ev == 'ext_bleu_ub':
            bleu = test_ext_bleu_upper_bound(test_dataset, types=[1, 2, 4])
            print('Extraction BLEU  Upper Bound: ', bleu)

        elif ev == 'ppl':
            ppl = test_exp_perplexity(test_dataset, model)
            print('Generation Perplexity:', ppl)

        elif ev == 'rmse':
            rmse = test_rate_rmse(test_dataset, model)
            print('RMSE:', rmse)

        else:
            greedy = args.search == 'greedy'
            topk = args.topk

            if ev.startswith('ext_'):
                ev = ev.replace('ext_', '')
                if has_extractor:
                    model = model.extractor
                searcher = ExtractDecoder(model, greedy=greedy)
            else:
                searcher = SearchDecoder(model, max_length=config.MAX_LENGTH, greedy=greedy, topk=topk)

            if ev in ['bleu', 'idf_bleu']:
                use_idf = (ev == 'idf_bleu')

                bleu = test_bleu(test_dataset, get_builder(model_type), searcher, types=[1, 2, 4], use_idf=use_idf)
                print(f'Generation {"IDF-" if use_idf else ""}BLEU: ', bleu)

            elif ev == 'length':
                length, idf = test_length(test_dataset, get_builder(model_type), searcher)
                print(f'Average Generation length & IDF: ', length, idf)

            elif ev == 'diversity':
                rep_l, seq_rep_2, uniq = test_diversity(test_dataset, get_builder(model_type), searcher)
                print(f'rep/l, seq_rep_2, uniq: ', rep_l, seq_rep_2, uniq)

            elif ev == 'self_bleu':
                bleus = test_self_bleu(test_dataset, get_builder(model_type), searcher)
                print(f'self bleu: ', bleus)


if __name__ == '__main__':
    with torch.no_grad():
        main()

import os
import argparse
import torch

import config
from main import init, load_dataset
from src.search_decoder import SearchDecoder, BeamSearchDecoder, DebiasSearchDecoder
from src.evaluate import test_rate_rmse, test_rate_mae, test_review_perplexity, test_feature_pr, test_rate_ndcg, test_review_mse, test_review_ndcg, test_review_bleu, load_ndcg

DIR_PATH = os.path.dirname(__file__)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', help='model name to save/load checkpoints')
  parser.add_argument('-c', '--checkpoint')
  parser.add_argument('-g', '--generator')
  parser.add_argument('--gen_checkpoint')
  parser.add_argument('-p', '--pred_rates', action='store_true')
  parser.add_argument('--show_distribution', action='store_true')

  parser.add_argument('-s', '--search', default='none', choices=['greedy', 'sample', 'beam', 'debias', 'none'], help='decoding search method, only work for BLEU')
  parser.add_argument('--sample_len', default=float('inf'), type=int, help='Sample length, only work if --search=sample')
  parser.add_argument('--beam_width', default=10, type=int, help='Beam width, only work if --search=beam')
  parser.add_argument('--debias_model', default='lm', help='Debias model, only work if --search=debias')
  parser.add_argument('--debias_model_checkpoint', default='best')

  parser.add_argument('evals', nargs='+')
  args = parser.parse_args()

  model, misc = init(args.model, args.checkpoint)
  model.eval()

  test_dataset = load_dataset('test')

  ranker = model
  if args.generator:
    generator, misc = init(args.generator, args.gen_checkpoint)
    generator.eval()
    model = generator

  # if searcger specified
  searcher = None
  if args.search == 'greedy':
    searcher = SearchDecoder(model, max_length=config.MAX_LENGTH, greedy=True)
  elif args.search == 'sample':
    searcher = SearchDecoder(model, max_length=config.MAX_LENGTH, greedy=False, sample_length=args.sample_len, topk=5)
  elif args.search == 'beam':
    searcher = BeamSearchDecoder(model, beam_width=args.beam_width, max_length=config.MAX_LENGTH, mode='best')
  elif args.search == 'debias':
    debias_model = init(args.debias_model, args.debias_model_checkpoint)[0]

    searcher = DebiasSearchDecoder(model, debias_model, max_length=config.MAX_LENGTH, n_debias=10)

  # Eval metrics
  for ev in args.evals:
    if ev == 'rmse':
      mse = test_rate_rmse(test_dataset, model)
      print('Rate RMSE: ', mse)

    elif ev == 'mae':
      mse = test_rate_mae(test_dataset, model)
      print('Rate MAE: ', mse)

    elif ev == 'rvw_rmse':
      mse = test_review_mse(test_dataset, ranker, searcher=searcher, pred_rates=args.pred_rates)
      print('Review RMSE: ', mse)

    elif ev == 'bleu':
      bleu1, bleu2, bleu4, length = test_review_bleu(test_dataset, searcher, types=[1, 2, 4])
      print(f'Review BLEU (1, 2, 4 from {length} non-empty reviews): ', bleu1, bleu2, bleu4)

    elif ev == 'ppl':
      ppl = test_review_perplexity(test_dataset, model)
      print('Review Perplexity: ', ppl)

    elif ev == 'f_pr':
      precision, recall = test_feature_pr(test_dataset, searcher)
      print('Feature Precision: ', precision)
      print('Feature Recall: ', recall)

    elif ev == 'ndcg':
      ndcg_user_items = load_ndcg(config.NDCG_TEST_FILE)
      k = [3, 5, 10, 15]
      print('User size:', len(ndcg_user_items))

      uid, vals = next(iter(ndcg_user_items.items()))
      size = len(vals)
      p_size = len([iid for iid in vals if test_dataset.get_review(uid, iid)])

      ndcg, p_ndcg = test_rate_ndcg(test_dataset, model, ndcg_user_items, k=k)
      print(f'Rate NDCG({size}):', ndcg)
      print(f'Rate Pure NDCG({p_size}):', p_ndcg)

    elif ev == 'rvw_ndcg':
      ndcg_user_items = load_ndcg(config.NDCG_TEST_FILE)

      print('User size:', len(ndcg_user_items))

      uid, iids = next(iter(ndcg_user_items.items()))
      p_size = len([1 for iid in iids if test_dataset.get_review(uid, iid)])

      ndcg = test_review_ndcg(ranker, test_dataset, ndcg_user_items, searcher=searcher, pred_rates=args.pred_rates)

      print(f'Rate NDCG({p_size}):', ndcg)


if __name__ == '__main__':
  with torch.no_grad():
    main()

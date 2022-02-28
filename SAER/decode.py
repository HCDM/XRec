'''
Decode
'''

import argparse
import json

import torch
from torch.utils.data import DataLoader

import config
from main import init, load_dataset
from src.voc import voc
from src.data import Review, ReviewBuilder
from src.search_decoder import SearchDecoder, BeamSearchDecoder, MCSearchDecoder
from src.utils import AttrDict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def decode(searcher, batch_data, ranker=None, n_samples=3):
  batch_data.to(DEVICE)

  ratings = None
  if hasattr(searcher.model, 'rate'):
    ratings = searcher.model.rate(batch_data).tolist()

  # expand batch, 1st dim must be batch size
  batch_data = AttrDict({
    k: t.expand(n_samples, *t.size()).flatten(0, 1)
    for k, t in batch_data
  })

  search_result = searcher(batch_data)
  exps = search_result.words
  rvw_lens = search_result.rvw_lens

  x_ratings = None
  if ranker:
    x_ratings, _ = ranker(exps, rvw_lens=rvw_lens)
    x_ratings = x_ratings.view(n_samples, -1).transpose_(0, 1).tolist()

  exps = exps.view(exps.size(0), n_samples, -1).transpose_(0, 2).tolist()
  rvw_lens = rvw_lens.view(n_samples, -1).transpose_(0, 1).tolist()

  exps = [
    [
      ' '.join([
        voc[w_idx] for w_idx in exp[:l]
        if w_idx != voc.eos_idx
      ])
      for exp, l in zip(exp_samples, ls) # review
    ]
    for exp_samples, ls in zip(exps, rvw_lens) # batch
  ]

  return ratings, exps, x_ratings


def decode_cli(searcher, ranker=None, output=None, n_samples=3, pp=False):
  inp = ''
  batch_builder = ReviewBuilder(need_scores=False, need_content=False)

  print('Input format: [User ID] [Item ID]')
  print('Input \'quit\' or \'q\' to quit')

  while True:
    # get input sentence
    inp = input('> ')
    # check if it is quit case
    if inp == 'q' or inp == 'quit':
      break

    user, item = [int(t) for t in inp.split(' ')]
    data = batch_builder([Review(user, item, 0)])

    ratings, exps, x_ratings = decode(searcher, data, ranker=ranker, n_samples=n_samples)

    print('%.1f' % ratings[0])
    for i, exp in enumerate(exps[0]):
      if x_ratings:
        print(exp, f'{x_ratings[0][i]:9.1f}')
      else:
        print(exp)


def decode_dataset(test_data, searcher, ranker=None, output=None, n_samples=3, pp=False):
  batch_builder = ReviewBuilder(need_scores=False, need_content=False)
  testloader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=batch_builder)

  if output:
    o_file = open(output, 'w')

  for _, batch_data in enumerate(testloader):
    ratings, samples, sample_ratings = decode(searcher, batch_data, ranker=ranker, n_samples=n_samples)

    batch_size = len(samples)

    for i in range(batch_size):
      uid, iid = batch_data.users[i].item(), batch_data.items[i].item()
      review = test_data.get_review(uid, iid)
      entity = dict(
        user=uid,
        item=iid,
        pred_score=None,
        exps=samples[i]
      )

      if ratings:
        entity['pred_score'] = ratings[i]

      if sample_ratings:
        sorted_exp_tuples = sorted(zip(samples[i], sample_ratings[i]), key=lambda p: abs(p[1] - entity['pred_score']))
        entity['exps'], entity['exp_scores'] = [list(l) for l in zip(*sorted_exp_tuples)]

      if output:
        o_file.write(json.dumps(entity))
        o_file.write('\n')

      if pp:
        print('User:', entity['user'], 'Item:', entity['item'])
        print('Rating:', review.score)
        print('Pred rating: %.1f' % entity['pred_score'])
        print('GT Exp:')

        for exp in review.text:
          print(exp)

        print('\nGen Exp:')

        for k, exp in enumerate(entity['exps']):
          if 'exp_scores' in entity:
            exp_score = entity['exp_scores'][k]
            print(exp, f'{exp_score:9.1f}')
          else:
            print(exp)

        print('\n')

  if output:
    o_file.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', help='model name to load checkpoints')
  parser.add_argument('-c', '--checkpoint')

  parser.add_argument('-p', '--prettyprint', action='store_true')
  parser.add_argument('-o', '--output', default=None, help='file path to write output json')
  parser.add_argument('-n', '--n_samples', default=3, type=int, help='number of samples')

  parser.add_argument('-r', '--ranker', help='text ranker to rank generated content')
  parser.add_argument('--ranker_checkpoint', default='best')

  parser.add_argument('-s', '--search', default='greedy', choices=['greedy', 'sample', 'beam', 'mc'])
  parser.add_argument('--sample_len', default=float('inf'), type=int, help='Sample length, only work if --search=sample')
  parser.add_argument('--beam_width', default=10, type=int, help='Beam width, only work if --search=beam')

  parser.add_argument('--cli', action='store_true', help='interactive mode')

  args = parser.parse_args()

  model, _ = init(args.model, args.checkpoint)

  # Set dropout layers to eval mode
  model.eval()

  # init content ranker
  ranker = None
  if args.ranker:
    ranker, _ = init(args.ranker, args.ranker_checkpoint)
    ranker.eval()

  # init search decoder
  if args.search == 'greedy':
    searcher = SearchDecoder(model, voc, max_length=config.MAX_LENGTH, greedy=True)
  elif args.search == 'sample':
    searcher = SearchDecoder(model, voc, max_length=config.MAX_LENGTH, greedy=False, sample_length=args.sample_len, topk=10)
  elif args.search == 'beam':
    searcher = BeamSearchDecoder(model, voc, beam_width=args.beam_width, max_length=config.MAX_LENGTH, mode='best')
  elif args.search == 'mc':
    searcher = MCSearchDecoder(model, voc, max_length=config.MAX_LENGTH, ranker=ranker)

  if args.cli:
    decode_cli(searcher, ranker=ranker, output=args.output, n_samples=args.n_samples)
  else:
    test_dataset = load_dataset('test')

    decode_dataset(test_dataset, searcher, ranker=ranker, output=args.output, n_samples=args.n_samples, pp=args.prettyprint)


if __name__ == '__main__':
  with torch.no_grad():
    main()

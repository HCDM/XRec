import math
from datetime import datetime
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch import optim

from .loss import mask_nll_loss, bpr_loss, lambda_rank_loss, rank_hinge_loss
from .data import ReviewGroupDataLoader, basic_builder, ReviewBuilder
from .evaluate import test_rate_ndcg, test_review_bleu, test_rate_rmse
from .search_decoder import SearchDecoder, GumbelDecoder
from .voc import voc

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class AbstractTrainer:
  ''' Abstract Trainer Pipeline '''

  def __init__(
    self,
    model,
    ckpt_mng,
    batch_size=64,
    lr=.01,
    l2=0,
    clip=1.,
    patience=5,
    max_iters=None,
    save_every=5,
    grp_config=None
  ):
    self.model = model

    self.ckpt_mng = ckpt_mng

    self.batch_size = batch_size

    self.optimizer = optim.Adam(
      model.parameters(),
      lr=lr,
      weight_decay=l2
    )
    self.clip = clip

    # trained epochs
    self.trained_epoch = 0
    self.train_results = []
    self.val_results = []
    self.best_epoch = self._best_epoch()

    self.collate_fn = basic_builder

    self.patience = patience
    self.max_iters = float('inf') if max_iters is None else max_iters
    self.save_every = save_every

    self.ckpt_name = lambda epoch: str(epoch)

    self.grp_config = grp_config

    # self.optim_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

  def log(self, *args):
    '''formatted log output for training'''

    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}   ', *args)

  def resume(self, checkpoint):
    '''load checkpoint'''

    self.trained_epoch = checkpoint['epoch']
    self.train_results = checkpoint['train_results']
    self.val_results = checkpoint['val_results']
    self.optimizer.load_state_dict(checkpoint['opt'])
    self.best_epoch = self._best_epoch()

  def reset_epoch(self):
    self.trained_epoch = 0
    self.train_results = []
    self.val_results = []
    self.best_epoch = self._best_epoch()

  def run_batch(self, training_batch, val=False):
    '''
    Run a batch of any batch size with the model

    Inputs:
      training_batch: train data batch created by batch_2_seq
      val: if it is for validation, no backward & optim
    Outputs:
      result: tuple (loss, *other_stats) of numbers or element tensor
        loss: a loss tensor to optimize
        other_stats: any other values to accumulate
    '''

    pass

  def run_epoch(self, train_data, dev_data):
    trainloader = ReviewGroupDataLoader(train_data, collate_fn=self.collate_fn, grp_config=self.grp_config, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # maximum iteration per epoch
    iter_len = min(self.max_iters, len(trainloader))

    # culculate print every to ensure ard 5 logs per epoch
    PRINT_EVERY = 10 ** round(math.log10(iter_len / 5))

    while True:
      epoch = self.trained_epoch + 1

      self.model.train()
      results_sum = []

      for idx, training_batch in enumerate(trainloader):
        if idx >= iter_len:
          break

        # run a training iteration with batch
        training_batch.to(DEVICE)
        batch_result = self.run_batch(training_batch)
        if type(batch_result) != tuple:
          batch_result = (batch_result,)

        loss = batch_result[0]

        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients: gradients are modified in place
        if self.clip:
          _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Adjust model weights
        self.optimizer.step()

        # Accumulate results
        self._accum_results(results_sum, batch_result)

        # Print progress
        iteration = idx + 1
        if iteration % PRINT_EVERY == 0:
          print_result = self._sum_to_result(results_sum, iteration)
          self.log('Epoch {}; Iter: {} {:.1f}%; {};'.format(epoch, iteration, iteration / iter_len * 100, self._result_to_str(print_result)))

      epoch_result = self._sum_to_result(results_sum, iteration)
      self.train_results.append(epoch_result)

      # validation
      with torch.no_grad():
        self.model.eval()
        val_result = self.validate(dev_data)
        self.model.train()

      self.log('Validation; Epoch {}; {};'.format(epoch, self._result_to_str(val_result)))

      self.val_results.append(val_result)

      # new best if no prev best or the sort key is smaller than prev best's
      is_new_best = self.best_epoch is None or \
          self._result_sort_key(val_result) < self._result_sort_key(self.val_results[self.best_epoch-1])

      self._handle_ckpt(epoch, is_new_best)
      self.trained_epoch += 1

      if is_new_best:
        self.best_epoch = epoch

      # self.optim_scheduler.step()

      yield is_new_best

  def train(self, train_data, dev_data):
    patience = self.patience # end the function when reaching threshold

    epoch = self.trained_epoch + 1

    # Data loaders with custom batch builder
    self.log(f'Start training from epoch {epoch}...')

    run_epoch = self.run_epoch(train_data, dev_data)

    while patience:
      is_new_best = next(run_epoch)

      # if better than before, recover patience; otherwise, lose patience
      if is_new_best:
        patience = self.patience
      else:
        patience -= 1

    best_result = self.val_results[self.best_epoch-1]
    self.log('Training ends: best result {} at epoch {}'.format(self._result_to_str(best_result), self.best_epoch))

  def validate(self, dev_data):
    devloader = ReviewGroupDataLoader(dev_data, collate_fn=self.collate_fn, grp_config=self.grp_config, batch_size=self.batch_size, shuffle=False)

    results_sum = []

    for dev_batch in devloader:
      dev_batch.to(DEVICE)
      result = self.run_batch(dev_batch, val=True)
      if type(result) != tuple:
        result = (result,)

      # Accumulate results
      self._accum_results(results_sum, result)

    return self._sum_to_result(results_sum, len(devloader))

  def _result_to_str(self, epoch_result):
    ''' convert result list to readable string '''
    return 'Loss: {:.4f}'.format(epoch_result)

  def _sum_to_result(self, results_sum, length):
    '''
    Convert accumulated sum of results to epoch result
    by default return the average batch loss
    '''
    loss_sum = results_sum[0]
    return loss_sum / length

  def _accum_results(self, results_sum, batch_result):
    ''' accumulate batch result of run batch '''

    while len(results_sum) < len(batch_result):
      results_sum.append(0)
    for i, val in enumerate(batch_result):
      results_sum[i] += val.item() if torch.is_tensor(val) else val

  def _result_sort_key(self, result):
    ''' return the sorting value of a result, the smaller the better '''
    return result

  def _best_epoch(self):
    '''
    get the epoch of best result, smallest sort key value, from results savings when resumed from checkpoint
    '''

    best_val, best_epoch = math.inf, None

    for i, result in enumerate(self.val_results):
      val = self._result_sort_key(result)
      if val < best_val:
        best_val = val
        best_epoch = i + 1

    return best_epoch

  def _handle_ckpt(self, epoch, is_new_best):
    '''
    Always save a checkpoint for the latest epoch
    Remove the checkpoint for the previous epoch
    If the latest is the new best record, remove the previous best
    Regular saves are exempted from removes
    '''

    # save new checkpoint
    cp_name = self.ckpt_name(epoch)
    self.ckpt_mng.save(cp_name, {
      'epoch': epoch,
      'train_results': self.train_results,
      'val_results': self.val_results,
      'model': self.model.state_dict(),
      'opt': self.optimizer.state_dict()
    }, best=is_new_best)
    self.log('Save checkpoint:', cp_name)

    epochs_to_purge = []
    # remove previous non-best checkpoint
    prev_epoch = epoch - 1
    if prev_epoch != self.best_epoch:
      epochs_to_purge.append(prev_epoch)

    # remove previous best checkpoint
    if is_new_best and self.best_epoch:
        epochs_to_purge.append(self.best_epoch)

    for e in epochs_to_purge:
      if e % self.save_every != 0:
        cp_name = self.ckpt_name(e)
        self.ckpt_mng.delete(cp_name)
        self.log('Delete checkpoint:', cp_name)


class RankerTrainer(AbstractTrainer):
  ''' Trainer to train recommendation ranking model '''

  def __init__(self, *args, rank_loss_type=None, loss_lambda=None, **kargs):
    super().__init__(*args, **kargs)

    self.rank_loss_type = rank_loss_type
    self.loss_lambda = loss_lambda

    if rank_loss_type:
      self.rank_loss_fn = {
        'RankHinge': rank_hinge_loss,
        'BPR': bpr_loss,
        'LambdaRank': lambda_rank_loss
      }[rank_loss_type]

  def run_batch(self, batch_data, val=False):
    '''
    Outputs:
      loss: tensor, overall loss to optimize
    '''

    rate_lam, rank_lam = self.loss_lambda['rate'], self.loss_lambda['rank']

    # extract fields from batch & set DEVICE options
    scores = batch_data.scores
    pred_scores = self.model.rate(batch_data)

    if self.rank_loss_type:

      grp_size = batch_data.grp_size
      pred_scores, scores = (t.view(-1, grp_size) for t in (pred_scores, scores))

      # only apply mse to rated items
      rate_loss = mse_loss(pred_scores[:, 0], scores[:, 0])

      rank_loss = self.rank_loss_fn(pred_scores, scores)

    else:
      rate_loss = mse_loss(pred_scores, scores)
      rank_loss = 0

    loss = rate_lam * rate_loss + rank_lam * rank_loss

    return loss

  def validate(self, dev_data):
    if self.rank_loss_type == 'LambdaRank':
      # loss is pointless in LambdaRank
      val_result = 0.
    else:
      val_result = super().validate(dev_data)

    rmse = test_rate_rmse(dev_data, self.model, builder=self.collate_fn, batch_size=self.batch_size)
    ndcg, pure_ndcg = test_rate_ndcg(dev_data, self.model, builder=self.collate_fn, batch_size=self.batch_size // 16)
    return val_result, rmse, ndcg, pure_ndcg

  def _result_to_str(self, epoch_result):
    if type(epoch_result) == tuple:
      s = 'LOSS: {:.4f}; RMSE: {:.4f}; NDCG: {:.4f}; P_NDCG: {:.4f}'.format(*epoch_result)
    else:
      s = 'LOSS: {:.4f}'.format(epoch_result)

    return s

  def _result_sort_key(self, result):
    ''' MSE loss '''
    return result[1]

class ReviewTrainer(AbstractTrainer):
  ''' Trainer to train review generation model '''

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)

    self.collate_fn = ReviewBuilder(need_scores=False)

    model = self.model
    for module in [model.user_ebd, model.item_ebd, model.ui_mlp, model.rater]:
      for parameter in module.parameters():
        parameter.requires_grad = False

  def run_batch(self, batch_data, val=False):
    '''
    Inputs:
      batch_data: (users, items, scores, words, mask)
        users: (batch_size, grp_size)
        items: (batch_size, grp_size)
        scores: (batch_size, grp_size)
        words: (seq_size, batch_size * grp_size)
        mask: (seq_size, batch_size * grp_size)
    Outputs:
      loss: tensor, overall loss to optimize
      loss_sum: number, sum of the loss
      n_words: number of words
    '''
    words, mask = batch_data.words, batch_data.mask

    # concat sos at the top & remove eos at the bottom
    sos_var = torch.full((1, words.size(-1)), voc.sos_idx, dtype=torch.long, device=DEVICE)
    inp = torch.cat([sos_var, words[:-1]])

    output_dict = self.model.review(batch_data, inp)
    review_loss = mask_nll_loss(output_dict.output, words, mask)

    sen_l1_loss = output_dict.rate_gates.masked_select(mask).mean()
    review_loss += 5e-3 * sen_l1_loss

    n_words = mask.sum().item()

    return review_loss, review_loss.item() * n_words, n_words

  def train(self, train_data, dev_data):
    train_data, dev_data = train_data.rvw_subset(), dev_data.rvw_subset()
    return super().train(train_data, dev_data)

  def _sum_to_result(self, results_sum, length):
    loss_sum = results_sum[1]
    n_words = results_sum[2]
    return loss_sum / n_words

  def validate(self, dev_data):
    val_result = super().validate(dev_data)
    # TODO: replace hardcoded searcher
    bleu2, bleu4, _ = test_review_bleu(dev_data.random_subset(10 ** 4), SearchDecoder(self.model, voc, max_length=30, greedy=False, topk=10))
    return (val_result, bleu2, bleu4)

  def _result_to_str(self, epoch_result):
    if type(epoch_result) == tuple:
      loss, rest = epoch_result[0], epoch_result[1:]
    else:
      loss, rest = epoch_result, None

    s = 'Loss: {:.4f}'.format(loss)
    if rest:
      s += '; BLEU2: {:.4f}; BLEU4: {:.4f}'.format(*rest)

    return s

  def _result_sort_key(self, result):
    ''' review loss '''
    return result[0]


class MultiTaskTrainer(RankerTrainer):
  ''' Trainer to train multi-task model '''

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)

    self.collate_fn = ReviewBuilder()

  def run_batch(self, batch_data, val=False):
    '''
    Outputs:
      loss: tensor, overall loss to optimize
      rate_loss: number, recomm loss
      review_loss: number, review loss
      n_words
    '''
    review_lam = self.loss_lambda['review']

    words, mask = batch_data.words, batch_data.mask

    # concat sos at the top & remove eos at the bottom
    sos_var = torch.full((1, words.size(-1)), voc.sos_idx, dtype=torch.long, device=DEVICE)
    inp = torch.cat([sos_var, words[:-1]])

    review_output = self.model.review(batch_data, inp)

    rate_loss = super().run_batch(batch_data, val)

    review_loss = mask_nll_loss(review_output.output, words, mask)

    sen_l1_loss = review_output.rate_gates.masked_select(mask).mean()
    review_loss += 5e-3 * sen_l1_loss

    loss = review_lam * review_loss + rate_loss # rate & rank lambda have been applied in ranker trainer

    n_words = mask.sum().item()

    return loss, rate_loss.item(), review_loss.item() * n_words, n_words

  def validate(self, dev_data):
    val_result, rmse, ndcg, p_ndcg = super().validate(dev_data)
    # TODO: replace hardcoded searcher
    bleu2, bleu4, _ = test_review_bleu(dev_data.random_subset(10 ** 4), SearchDecoder(self.model, voc, max_length=30, greedy=False, topk=10))
    return (*val_result, rmse, ndcg, p_ndcg, bleu2, bleu4)

  def _sum_to_result(self, results_sum, length):
    _, rate_sum, review_sum, n_words = results_sum
    rate_loss = rate_sum / length
    review_loss = review_sum / n_words
    return rate_loss, review_loss

  def _result_to_str(self, epoch_result):
    s = 'Rate loss: {:.4f}; Review loss: {:.4f}'.format(*epoch_result[:2])
    if len(epoch_result) > 2:
      s += '; RMSE: {:4f}, NDCG: {:.4f}; P_NDCG: {:.4f}; BLEU2: {:.4f}; BLEU4: {:.4f}'.format(*epoch_result[2:])

    return s

  def _result_sort_key(self, result):
    ''' RMSE '''
    return result[2]


class SentimentRegressorTrainer(RankerTrainer):
  ''' Trainer to train review ranker model '''

  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)

    self.collate_fn = ReviewBuilder()

  def train(self, train_data, dev_data):
    train_data, dev_data = train_data.rvw_subset(), dev_data.rvw_subset()
    return super().train(train_data, dev_data)

  def run_batch(self, batch_data, val=False):
    '''
    Inputs:
      batch_data: (users, items, scores, words, mask)
        users: (batch_size, grp_size)
        items: (batch_size, grp_size)
        scores: (batch_size, grp_size)
        words: (seq_size, batch_size * grp_size)
        mask: (seq_size, batch_size * grp_size)
    Outputs:
      loss: tensor, overall loss to optimize
    '''

    # extract fields from batch & set DEVICE options
    scores, words, mask = batch_data.scores, batch_data.words, batch_data.mask

    rvw_rank_output, _ = self.model(words, word_mask=mask)

    rank_loss = mse_loss(rvw_rank_output, scores)

    return rank_loss

  def _result_to_str(self, epoch_result):
    if type(epoch_result) == tuple:
      loss, ndcg = epoch_result
    else:
      loss, ndcg = epoch_result, None

    s = 'Loss: {:.4f}'.format(loss)
    if ndcg:
      s += '; NDCG: {:.4f}'.format(ndcg)

    return s

  def _result_sort_key(self, result):
    ''' MSE loss '''
    return result


class GANGeneratorTrainer(AbstractTrainer):
  ''' Trainer to train review generation model in GAN '''
  def __init__(self, model, *args, ranker=None, discriminator=None, max_length=0, loss_lambda=None, **kargs):

      super().__init__(model, *args, **kargs)

      self.ranker = ranker

      self.gumble_searcher = GumbelDecoder(model, voc, max_length=max_length)
      self.sample_searcher = SearchDecoder(model, voc, max_length=max_length, greedy=False, topk=10)

      self.discriminator = discriminator

      self.max_length = max_length
      self.loss_lambda = loss_lambda

      self.collate_fn = ReviewBuilder()

  def freeze(self):
    for parameter in self.model.parameters():
      parameter.requires_grad = False

  def defreeze(self):
    for parameter in self.model.parameters():
      parameter.requires_grad = True

    model = self.model
    for module in [model.user_ebd, model.item_ebd, model.ui_mlp, model.rater]:
      for parameter in module.parameters():
        parameter.requires_grad = False

  def run_batch(self, batch_data, val=False):
    '''
    Inputs:
      batch_data: (users, items, scores)
        users: (batch_size, grp_size)
        items: (batch_size, grp_size)
        scores: (batch_size, grp_size)
    Outputs:
      loss: tensor, overall loss to optimize
      loss_sum: number, sum of the loss
      n_words: number of words
    '''
    alpha, beta, gamma = self.loss_lambda['rate'], self.loss_lambda['gan'], self.loss_lambda['nll']

    if not val:
      search_result = self.gumble_searcher(batch_data)
    else:
      search_result = self.sample_searcher(batch_data)

    search_words, rvw_lens = search_result.words, search_result.rvw_lens
    rate_output = self.model.rate(batch_data)

    # GAN loss
    logits = self.discriminator(search_words, rvw_lens=rvw_lens)

    labels = torch.ones(logits.size(), device=logits.device)
    gan_loss = binary_cross_entropy_with_logits(logits, labels)

    # Review rating loss
    rating_output, _ = self.ranker(search_words, rvw_lens=rvw_lens)
    rating_loss = mse_loss(rating_output, rate_output)

    # NLL loss
    words, mask = batch_data.words, batch_data.mask

    sos_var = torch.full((1, words.size(-1)), voc.sos_idx, dtype=torch.long, device=DEVICE)
    inp = torch.cat([sos_var, words[:-1]])

    output_dict = self.model.review(batch_data, inp)
    nll_loss = mask_nll_loss(output_dict.output, words, mask)

    # sen gate lasso
    sen_l1_loss = output_dict.rate_gates.masked_select(mask).mean()

    # All loss
    loss = alpha * rating_loss + beta * gan_loss + gamma * (nll_loss + 5e-3 * sen_l1_loss)

    return loss, rating_loss, gan_loss, nll_loss

  def validate(self, dev_data):
    val_result = super().validate(dev_data)

    bleu2, bleu4, _ = test_review_bleu(dev_data.random_subset(10 ** 4), self.sample_searcher, voc)

    return (*val_result, bleu2, bleu4)

  def _result_to_str(self, epoch_result):
    s = 'Loss: {:.4f}; MSE Loss: {:.4f}; GAN Loss: {:.4f}; NLL Loss: {:.4f};'.format(*epoch_result[:4])

    if len(epoch_result) > 4:
      s += '; BLEU2: {:.4f}; BLEU4: {:.4f}'.format(*epoch_result[4:])

    return s

  def _sum_to_result(self, results_sum, length):
    return tuple(l / length for l in results_sum)

  def _result_sort_key(self, result):
    return -result[4] # BLEU2


class GANDiscriminatorTrainer(AbstractTrainer):
  ''' Trainer to train review discriminator model in GAN '''
  def __init__(self, model, *args, generator=None, max_length=0, **kargs):

      super().__init__(model, *args, **kargs)

      # for module in [model.user_ebd, model.item_ebd, model.rater]:
      #   for parameter in module.parameters():
      #     parameter.requires_grad = False

      self.searcher = SearchDecoder(generator, voc, max_length=max_length, greedy=False)
      # self.searcher = GumbelDecoder(generator, voc, max_length=max_length)

      self.max_length = max_length

      self.collate_fn = ReviewBuilder(no_empty=True)

  def freeze(self):
    for parameter in self.model.parameters():
      parameter.requires_grad = False

  def defreeze(self):
    for parameter in self.model.parameters():
      parameter.requires_grad = True

  def run_batch(self, batch_data, val=False):
    '''
    Inputs:
      batch_data: (users, items, scores)
        users: (batch_size, grp_size)
        items: (batch_size, grp_size)
        scores: (batch_size, grp_size)
    Outputs:
      loss: tensor, overall loss to optimize
      loss_sum: number, sum of the loss
      n_words: number of words
    '''

    logits = self.model(batch_data.words, word_mask=batch_data.mask)
    pos_labels = torch.ones(logits.size(), device=logits.device)
    pos_loss = binary_cross_entropy_with_logits(logits, pos_labels)

    for _ in range(1):
      search_dict = self.searcher(batch_data)

      neg_logits = self.model(search_dict.words, rvw_lens=search_dict.rvw_lens)
      neg_labels = torch.zeros(neg_logits.size(), device=logits.device)

      neg_loss = binary_cross_entropy_with_logits(neg_logits, neg_labels)

    loss = (pos_loss + neg_loss) / 2
    return loss, pos_loss.item(), neg_loss.item()

  def _result_to_str(self, epoch_result):
    loss, pos_loss, neg_loss = epoch_result
    s = 'Loss: {:.4f}; Pos Loss: {:.4f}; Neg Loss: {:.4f}'.format(loss, pos_loss, neg_loss)

    return s

  def _sum_to_result(self, results_sum, length):
    return tuple(l / length for l in results_sum)

  def _result_sort_key(self, result):
    return result[0]


class GANTrainer(AbstractTrainer):
  def __init__(self, generator, gen_ckp_mng, discriminator, dis_ckp_mng, max_length=15, patience=5, save_every=5, gen_config=None, dis_config=None):
    self.generator = generator
    self.discriminator = discriminator

    self.gen_trainer = GANGeneratorTrainer(generator, gen_ckp_mng, discriminator=discriminator, max_length=max_length, **gen_config)
    self.dis_trainer = GANDiscriminatorTrainer(discriminator, dis_ckp_mng, generator=generator, max_length=max_length, **dis_config)

    self.patience = patience
    self.save_every = save_every

  def resume(self, checkpoint):
    '''load checkpoint'''

    gen_ckpt, dis_ckpt = checkpoint
    if gen_ckpt:
      self.gen_trainer.resume(gen_ckpt)
    if dis_ckpt:
      self.dis_trainer.resume(dis_ckpt)

  def reset_epoch(self):
    self.gen_trainer.reset_epoch()
    self.dis_trainer.reset_epoch()

  def train(self, train_data, dev_data):
    patience = self.patience  # end the function when reaching threshold

    # Data loaders with custom batch builder
    train_data, dev_data = train_data.rvw_subset(), dev_data.rvw_subset()
    gen_run_epoch = self.gen_trainer.run_epoch(train_data, dev_data)
    dis_run_epoch = self.dis_trainer.run_epoch(train_data, dev_data)

    while patience:
      dis_epoch = self.dis_trainer.trained_epoch + 1
      gen_epoch = self.gen_trainer.trained_epoch + 1

      if dis_epoch < gen_epoch + 2:
        self.log(f'Training discriminator from epoch {dis_epoch}...')

        self.gen_trainer.freeze()
        self.dis_trainer.defreeze()

        _ = next(dis_run_epoch)

      if dis_epoch < 2:
        continue

      self.log(f'Training generator from epoch {gen_epoch}...')

      self.gen_trainer.defreeze()
      self.dis_trainer.freeze()

      is_gen_new_best = next(gen_run_epoch)

      # if better than before, recover patience; otherwise, lose patience
      if is_gen_new_best:
        patience = self.patience
      else:
        patience -= 1

    best_result = self.gen_trainer.val_results[self.gen_trainer.best_epoch-1]
    self.log('Training ends: generators best result {} at epoch {}'.format(self.gen_trainer._result_to_str(best_result), self.gen_trainer.best_epoch))


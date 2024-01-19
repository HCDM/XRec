import math
from datetime import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import DEVICE, MAX_LENGTH
from .dataset import ExtBuilder, BleuRankBuilder, CompExpGenBuilder
from .voc import voc
from .loss import mask_nll_loss
from .decoder import SearchDecoder
from .utils import ParallelBleu


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

        self.collate_fn = None  # must be rewritten

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
        trainloader = DataLoader(train_data, collate_fn=self.collate_fn,
                                 batch_size=self.batch_size, shuffle=True, num_workers=4)

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
                    _ = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip)

                # Adjust model weights
                self.optimizer.step()

                # Accumulate results
                self._accum_results(results_sum, batch_result)

                # Print progress
                iteration = idx + 1
                if iteration % PRINT_EVERY == 0:
                    print_result = self._sum_to_result(results_sum, iteration)
                    self.log('Epoch {}; Iter: {} {:.1f}%; {};'.format(
                        epoch, iteration, iteration / iter_len * 100, self._result_to_str(print_result)))

            epoch_result = self._sum_to_result(results_sum, iteration)
            self.train_results.append(epoch_result)

            # validation
            with torch.no_grad():
                self.model.eval()
                val_result = self.validate(dev_data)
                self.model.train()

            self.log('Validation; Epoch {}; {};'.format(
                epoch, self._result_to_str(val_result)))

            self.val_results.append(val_result)

            # new best if no prev best or the sort key is smaller than prev best's
            is_new_best = self.best_epoch is None or \
                self._result_sort_key(val_result) < self._result_sort_key(
                    self.val_results[self.best_epoch-1])

            self._handle_ckpt(epoch, is_new_best)
            self.trained_epoch += 1

            if is_new_best:
                self.best_epoch = epoch

            # self.optim_scheduler.step()

            yield is_new_best

    def train(self, train_data, dev_data):
        patience = self.patience  # end the function when reaching threshold

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
        self.log('Training ends: best result {} at epoch {}'.format(
            self._result_to_str(best_result), self.best_epoch))

    def validate(self, dev_data):
        devloader = DataLoader(
            dev_data, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False)

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


class ExtractorTrainer(AbstractTrainer):
    ''' Trainer to train ranking model '''

    def __init__(self, *args, n_item_exps=None, n_ref_exps=None, n_pos_exps=None, n_user_exps=None, **kargs):
        super().__init__(*args, **kargs)

        # self.collate_fn = BleuExtBuilder(
        self.collate_fn = ExtBuilder(
            n_item_exps=n_item_exps, n_ref_exps=n_ref_exps, n_pos_exps=n_pos_exps, n_user_exps=n_user_exps)

    def run_batch(self, batch_data, val=False):
        '''
        Outputs:
          loss: tensor, overall loss to optimize
        '''

        labels = batch_data.item_exp_label
        result = self.model.extractor(batch_data)  # (batch, n_item_exps)
        probs = result.probs

        pos_probs = probs[labels]
        loss = - (pos_probs + 1e-10).log().mean()

        return loss


class CompExpTrainer(AbstractTrainer):
    ''' Trainer to train rewrite generator '''

    def __init__(self, *args, rewrite_dataset, n_item_exps=None, n_ref_exps=None, n_pos_exps=None, n_user_exps=None, ext_loss='NLL', loss_lambda=None, use_idf=False, **kargs):
        super().__init__(*args, **kargs)

        self.loss_lambda = loss_lambda

        self.rewrite_collate_fn = CompExpGenBuilder(
            rvw_data=None, n_ref_exps=n_ref_exps)

        self.ext_loss = ext_loss
        if ext_loss == 'NLL':
            self.collate_fn = ExtBuilder(
                n_item_exps=n_item_exps, n_ref_exps=n_ref_exps, n_pos_exps=n_pos_exps, n_user_exps=n_user_exps)
        elif ext_loss == 'BLEU':
            self.collate_fn = BleuRankBuilder(
                n_item_exps=n_item_exps, n_ref_exps=n_ref_exps, n_pos_exps=n_pos_exps, n_user_exps=n_user_exps, bleu_type=2, adv=False, use_idf=use_idf)

        self.rewrite_loader = DataLoader(
            rewrite_dataset, collate_fn=self.rewrite_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.rewrite_iter = None

    def train(self, train_data, dev_data):
        self.rewrite_collate_fn.rvw_data = train_data
        self.rewrite_iter = iter(self.rewrite_loader)
        return super().train(train_data, dev_data)

    def run_batch(self, batch_data, val=False):
        '''
        Outputs:
          loss: tensor, overall loss to optimize
        '''
        labels = batch_data.item_exp_label
        result = self.model.extractor(batch_data)  # (batch, n_item_exps)
        probs = result.probs

        if self.ext_loss == 'NLL':
            pos_probs = probs[labels]
            ext_loss = - (pos_probs + 1e-10).log().mean()
        elif self.ext_loss == 'BLEU':
            ext_loss = - (probs * labels).sum(1).mean()

        loss = ext_loss

        # rewrite
        rewrite_loss = 0
        if not val:
            rewrite_batch_data = next(self.rewrite_iter, None)
            if not rewrite_batch_data:
                self.rewrite_iter = iter(self.rewrite_loader)
                rewrite_batch_data = next(self.rewrite_iter, None)

            rewrite_batch_data.to(DEVICE)
            words = rewrite_batch_data.words

            # concat sos at the top & remove eos at the bottom
            sos_var = torch.full((words.size(0), 1), voc.sos_idx,
                                 dtype=torch.long, device=words.device)
            inp = torch.cat([sos_var, words[..., :-1]], dim=1)

            # (batch, n_item_exps)
            gen_results = self.model(inp, rewrite_batch_data)
            log_probs = gen_results.output
            rewrite_loss = mask_nll_loss(
                log_probs, words, rewrite_batch_data.words_mask)

            loss = self.loss_lambda['ext'] * loss + \
                self.loss_lambda['gen'] * rewrite_loss

            # loss += 2 * result.energies.mean()

        return loss, ext_loss, rewrite_loss

    def _result_to_str(self, epoch_result):
        ''' convert result list to readable string '''
        return 'Loss: {:.4f}; Ext Loss: {:.4f}; Gen Loss: {:.4f};'.format(*epoch_result)

    def _sum_to_result(self, results_sum, length):
        '''
        Convert accumulated sum of results to epoch result
        by default return the average batch loss
        '''
        return tuple(loss_sum / length for loss_sum in results_sum)


class CompExpPolicyGradientTrainer(AbstractTrainer):
    ''' Trainer to train rewrite generator '''

    def __init__(self, *args, rewrite_dataset=None, n_item_exps=None, n_ref_exps=None, use_idf=False, bleu_weight=None, loss_lambda=None, **kargs):
        super().__init__(*args, **kargs)

        self.model.ext_policy = 'sample'
        self.searcher = SearchDecoder(
            self.model, greedy=False, max_length=MAX_LENGTH)
        self.pb = ParallelBleu(8)

        self.rewrite_loader, self.rewrite_iter = None, None
        if rewrite_dataset:
            print('with rewrite')
            self.rewrite_collate_fn = CompExpGenBuilder(
                rvw_data=None, n_ref_exps=n_ref_exps)

            self.rewrite_loader = DataLoader(
                rewrite_dataset, collate_fn=self.rewrite_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

        self.collate_fn = ExtBuilder(
            n_item_exps=n_item_exps, n_ref_exps=n_ref_exps, n_pos_exps=0, n_user_exps=0, return_rvws=True)

        self.loss_lambda = loss_lambda
        self.use_idf = use_idf

        self.bleu_weight = bleu_weight

    def train(self, train_data, dev_data):
        if self.rewrite_loader:
            self.rewrite_collate_fn.rvw_data = train_data
            self.rewrite_iter = iter(self.rewrite_loader)

        return super().train(train_data, dev_data)

    def run_batch(self, batch_data, val=False):
        '''
        Outputs:
          loss: tensor, overall loss to optimize
        '''
        # item_exp_bleus = batch_data.item_exp_label
        ext_log_prob_list, gen_log_prob_list = [], []
        ext_bleu_list, gen_bleu_list, ext_recall_list = [], [], []

        for _ in range(5 if not val else 1):
            result = self.searcher(batch_data)  # (batch, n_item_exps)
            ext_probs, ext_indices = result.ext_probs, result.ext_indices
            exted_probs = ext_probs[range(ext_indices.size(0)), ext_indices]
            log_exted_probs = (exted_probs + 1e-10).log()

            gen_words, gen_probs, gen_lens = result.words, result.probs, result.words_lens
            log_gen_porbs = gen_probs.log()
            for i, l in enumerate(gen_lens):
                log_gen_porbs[i, l:] = 0
            log_gen_porbs = log_gen_porbs.sum(-1)

            ext_log_prob_list.append(log_exted_probs)
            gen_log_prob_list.append(log_gen_porbs)

            # Reward advantage
            # base_bleus = (ext_probs * item_exp_bleus).sum(1)

            exps = [
                [voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx]
                for exp, l in zip(gen_words, gen_lens)
            ]
            refs = [rvw.text for rvw in batch_data.rvws]

            bleus = self.pb(exps, refs, use_idf=self.use_idf,
                            weights=self.bleu_weight)
            bleus = torch.tensor([b[0] for b in bleus],
                                 device=log_gen_porbs.device)
            gen_bleu_list.append(bleus)

            exts = [item_exps[idx]
                    for item_exps, idx in zip(batch_data.item_exps, ext_indices)]

            ext_recall = self.pb.ext_recall(
                exts, exps, refs, weights=(.7, .3), use_idf=self.use_idf)
            ext_recall = torch.tensor(
                [r[0] for r in ext_recall], device=log_gen_porbs.device)
            ext_recall_list.append(ext_recall)

            exted_bleus = self.pb(
                exts, refs, weights=self.bleu_weight, use_idf=self.use_idf)
            exted_bleus = torch.tensor(
                [b[0] for b in exted_bleus], device=log_gen_porbs.device)
            ext_bleu_list.append(exted_bleus)

        ext_log_probs = torch.stack(ext_log_prob_list, dim=1)
        gen_log_probs = torch.stack(gen_log_prob_list, dim=1)
        ext_bleu = torch.stack(ext_bleu_list, dim=1)
        gen_bleu = torch.stack(gen_bleu_list, dim=1)
        ext_recall = torch.stack(ext_recall_list, dim=1)

        ext_rewards = gen_bleu + 0.2 * ext_bleu
        gen_rewards = gen_bleu + 0.2 * ext_recall

        ext_adv = ext_rewards - ext_rewards.mean(-1, keepdim=True)
        gen_adv = gen_rewards - gen_rewards.mean(-1, keepdim=True)

        loss = - (ext_adv * ext_log_probs).mean() - \
            (gen_adv * gen_log_probs).mean()

        # rewrite
        if not val and self.rewrite_iter:
            rewrite_batch_data = next(self.rewrite_iter, None)
            if not rewrite_batch_data:
                self.rewrite_iter = iter(self.rewrite_loader)
                rewrite_batch_data = next(self.rewrite_iter, None)

            rewrite_batch_data.to(DEVICE)
            words = rewrite_batch_data.words

            # concat sos at the top & remove eos at the bottom
            sos_var = torch.full((words.size(0), 1), voc.sos_idx,
                                 dtype=torch.long, device=words.device)
            inp = torch.cat([sos_var, words[..., :-1]], dim=1)

            # (batch, n_item_exps)
            results = self.model(inp, rewrite_batch_data)
            log_probs = results.output
            rewrite_loss = mask_nll_loss(
                log_probs, words, rewrite_batch_data.words_mask)

            loss = self.loss_lambda['pg'] * loss + \
                self.loss_lambda['gen'] * rewrite_loss

        return loss, gen_bleu.mean()

    def _result_to_str(self, epoch_result):
        ''' convert result list to readable string '''
        return 'Loss: {:.4f}; BLEU: {:.4f};'.format(*epoch_result)

    def _result_sort_key(self, result):
        ''' return the sorting value of a result, the smaller the better '''
        return -result[1]

    def _sum_to_result(self, results_sum, length):
        '''
        Convert accumulated sum of results to epoch result
        by default return the average batch loss
        '''
        return tuple(loss_sum / length for loss_sum in results_sum)

import argparse

import config
from main import init, load_dataset, load_rewrite_dataset
from src.trainer import CompExpTrainer, CompExpPolicyGradientTrainer


KARGS_LOG_KEYS = {'batch_size', 'lr', 'l2', 'clip', 'rank_loss_type',
                  'loss_lambda', 'max_iters', 'max_length', 'n_item_exps', 'n_ref_exps', 'n_pos_exps', 'n_user_exps', 'ext_loss', 'use_idf', 'bleu_weight'}


def config_to_kargs(model_config):
    return dict(
        batch_size=model_config.BATCH_SIZE,
        lr=model_config.LR,
        l2=model_config.L2_PENALTY,
        clip=model_config.CLIP,
        patience=config.PATIENCE,
        max_iters=model_config.MAX_ITERS,
        save_every=config.SAVE_EVERY,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='model name to save/load checkpoints')
    parser.add_argument('-c', '--checkpoint')

    args = parser.parse_args()

    model, misc = init(args.model, args.checkpoint)
    checkpoint, ckpt_mng, model_config = (
        misc[k] for k in ['checkpoint', 'ckpt_mng', 'model_config']
    )

    trainer_args = (model, ckpt_mng)

    kargs = config_to_kargs(model_config)

    if model_config.TRAINING_TASK == 'comp_exp':
        Trainer = CompExpTrainer

        kargs['rewrite_dataset'] = load_rewrite_dataset()
        kargs['n_item_exps'] = model_config.N_ITEM_EXPS
        kargs['n_ref_exps'] = model_config.N_REF_EXPS
        kargs['n_pos_exps'] = model_config.N_POS_EXPS
        kargs['n_user_exps'] = model_config.N_USER_EXPS
        kargs['ext_loss'] = model_config.EXT_LOSS
        kargs['use_idf'] = model_config.USE_IDF

        kargs['loss_lambda'] = model_config.LOSS_LAMBDA

    elif model_config.TRAINING_TASK == 'pg':
        Trainer = CompExpPolicyGradientTrainer

        if model_config.REWRITE:
            rewrite_dataset = load_rewrite_dataset()
            kargs['rewrite_dataset'] = rewrite_dataset
        kargs['use_idf'] = model_config.USE_IDF
        kargs['bleu_weight'] = model_config.BLEU_WEIGHT

        kargs['n_item_exps'] = model_config.N_ITEM_EXPS
        kargs['n_ref_exps'] = model_config.N_REF_EXPS

        kargs['loss_lambda'] = model_config.LOSS_LAMBDA

    print(f'Training method: {model_config.TRAINING_TASK}')
    print(f'Training config:', {k: v for k,
                                v in kargs.items() if k in KARGS_LOG_KEYS})

    trainer = Trainer(
        *trainer_args,
        **kargs
    )

    if checkpoint:
        trainer.resume(checkpoint)

    train_dataset = load_dataset('train')
    dev_dataset = load_dataset('dev')

    model.train()

    trainer.train(train_dataset, dev_dataset)


if __name__ == '__main__':
    main()

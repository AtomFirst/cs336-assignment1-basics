import os
import yaml
import argparse
from functools import partial
from pathlib import Path
import copy

import wandb

from .trainer import Trainer, WANDB_CONFIG


def lr_tuning(train_config):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'valid_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 51200 // train_config['train']['batch_size'],
            'eta': 2,
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
        }
    }

    sweep_id = wandb.sweep(
        **WANDB_CONFIG,
        sweep=sweep_config
    )

    def update_nested_config(
        config: dict[str, dict],
        sweep_config: dict
    ):
        epochs = config['train']['epochs']
        
        config['lr_schedule'].update({
            'max_learning_rate': sweep_config['lr'],
            'min_learning_rate': sweep_config['lr'] * 0.1,
            'warmup_iters': int(epochs * 0.05),
            'cosine_cycle_iters': int(epochs * 0.9),
        })
        
        config['checkpoint']['path'] = None

        return config

    def sweep_train(base_config):
        with wandb.init() as run:
            config = update_nested_config(
                copy.deepcopy(base_config),
                dict(run.config),
            )

            trainer = Trainer(config)
            trainer.train(run)

    wandb.agent(
        sweep_id,
        function=partial(sweep_train, train_config),
        count=5,
    )


def test(train_config):
    total_batch = 1_280_000
    for batch_size in [1, 16, 64, 256, 512]:
        train_config['train'].update({
            'epochs': total_batch // batch_size,
            'batch_size': batch_size,
        })

        lr_tuning(train_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    test(config)
    

if __name__ == '__main__':
    main()

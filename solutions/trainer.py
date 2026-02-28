import os
import yaml
import argparse
from functools import partial
from pathlib import Path
import copy

import numpy as np
import torch
import torch.optim as optim
import wandb

from implements.transformer import TransformerLM
import implements.training as trn
import implements.training_loop as trnlp


class Trainer:
    def __init__(self, config):
        self.config = config

        model_cfg = self.config['model']
        device_cfg = self.config['device']
        self.model = TransformerLM(**model_cfg, device=device_cfg)

        optimizer_cfg = self.config['optimizer']
        betas = tuple(optimizer_cfg['betas'])
        self.optimizer = trn.AdamW(self.model.parameters(), lr=1.0, betas=betas)

        lr_cfg = self.config['lr_schedule']
        self.lr_cosine_schedule = partial(trn.get_lr_cosine_schedule, **lr_cfg)

        data_cfg = self.config['datasets']

        def memmap(file_path, dtype):
            file_size_bytes = os.path.getsize(file_path)
            num_elements = file_size_bytes // np.dtype(dtype).itemsize
            return np.memmap(file_path, dtype=dtype, mode='r', shape=(num_elements,))
        
        self.train_dataset = memmap(data_cfg['train'], np.uint16)
        self.valid_dataset = memmap(data_cfg['valid'], np.uint16)

        train_cfg = self.config['train']
        self.get_batch = partial(trnlp.get_batch, batch_size=train_cfg['batch_size'], context_length=model_cfg['context_length'], device=device_cfg)

        self.iteration = 0

        checkpoint_path = self.config['checkpoint']['path']
        if checkpoint_path is None:
            print('\ncheckpoint_path is None!\n')
        elif Path(checkpoint_path).exists():
            self.iteration = trnlp.load_checkpoint(checkpoint_path, self.model, self.optimizer)
    
    def save_checkpoint(self, path: str | None = None):
        path = path or self.config['checkpoint']['path']
        if path:
            trnlp.save_checkpoint(self.model, self.optimizer, self.iteration, path)

    def train(self, run: wandb.Run | None = None, epochs: int | None = None):
        epochs = epochs or self.config['train']['epochs']
        batch_size = self.config['train']['batch_size']

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_cosine_schedule, self.iteration - 1)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            inputs, targets = self.get_batch(self.train_dataset)
            inputs = inputs.int()
            targets = targets.int()

            outputs = self.model(inputs)
            loss = trn.cross_entropy(outputs, targets)
            loss.backward()

            valid_loss = self.evaluate()

            if run is not None:
                run.log({
                    'train_loss': loss,
                    'valid_loss': valid_loss,
                    'iteration': self.iteration,
                    'batch': self.iteration * batch_size,
                })

            if (epoch + 1) % self.config['train']['saving_per_epochs'] == 0:
                self.save_checkpoint()

            self.optimizer.step()
            scheduler.step()
            self.iteration += 1

        self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self, epochs: int = 1) -> float:
        self.model.eval()
        losses = []

        for _ in range(epochs):
            inputs, targets = self.get_batch(self.valid_dataset)
            inputs = inputs.int()
            targets = targets.int()

            outputs = self.model(inputs)
            loss = trn.cross_entropy(outputs, targets)
            losses.append(loss)

        losses = torch.stack(losses)
        mean_loss = torch.mean(losses).item()

        return mean_loss


WANDB_CONFIG = {
    "entity": "3235965152-nanjing-university-of-aeronautics-and-astrona",
    "project": "cs336-assignment1-experiment",    
}


def train(config):
    run = wandb.init(
        **WANDB_CONFIG,
        config=config,
    )

    trainer = Trainer(config)
    trainer.train(run)

    run.finish()


def lr_tuning(config):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'valid_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 200,
            'eta': 2,
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
        }
    }

    sweep_id = wandb.sweep(
        **WANDB_CONFIG,
        sweep=sweep_config, 
    )

    def update_nested_config(
        config: dict[str, dict],
        sweep_config: dict
    ):
        total_batch = 1_280_000
        # epochs = total_batch // sweep_config['batch_size']
        epochs = config['train']['epochs']
        
        config['lr_schedule'].update({
            'max_learning_rate': sweep_config['lr'],
            'min_learning_rate': sweep_config['lr'] * 0.1,
            'warmup_iters': int(epochs * 0.05),
            'cosine_cycle_iters': int(epochs * 0.9),
        })

        # config['train'].update({
        #     'epochs': epochs,
        #     'batch_size': sweep_config['batch_size'],
        # })
        
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
        function=partial(sweep_train, config),
        count=10,
    )


def test(config):
    total_batch = 1_280_000
    for batch_size in [1, 16, 128, 512]:
        config['train'].update({
            'epochs': total_batch // batch_size,
            'batch_size': batch_size,
        })
        
        lr_tuning(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.test:
        test(config)
    else:
        train(config)


if __name__ == '__main__':
    main()

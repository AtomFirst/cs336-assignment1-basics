import os
import yaml
import argparse
import logging
from functools import partial
from typing import BinaryIO, IO

import numpy as np
import torch
import torch.optim as optim

from implements.transformer import TransformerLM
import implements.training as trn
import implements.training_loop as trnlp


logger = logging.getLogger("training")
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('train.log')

format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(format)
f_handler.setFormatter(format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


'''
need conf:
TransformerLM
trn.AdamW
trn.get_lr_cosine_schedule
trnlp.get_batch
'''
class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['model']
        self.model = TransformerLM(**model_cfg, device=self.config['train']['device'])

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
        self.vaild_dataset = memmap(data_cfg['vaild'], np.uint16)

        batch_cfg = self.config['train']
        self.batch = partial(trnlp.get_batch, **batch_cfg, context_length=model_cfg['context_length'])

        self.iteration = 0

        checkpoint_path = self.config['checkpoint']['path']
        if checkpoint_path is not None:
            self.iteration = trnlp.load_checkpoint(checkpoint_path, self.model, self.optimizer)
    
    def save_checkpoint(self, out: str | os.PathLike | BinaryIO | IO[bytes]):
        trnlp.save_checkpoint(self.model, self.optimizer, self.iteration, out)

    def train(self, epochs: int):
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_cosine_schedule, self.iteration - 1)

        for _ in range(epochs):
            self.optimizer.zero_grad()

            inputs, targets = self.batch(self.train_dataset)
            inputs = inputs.int()
            targets = targets.int()

            outputs = self.model(inputs)
            loss = trn.cross_entropy(outputs, targets)
            loss.backward()

            if self.iteration % 100 == 0:
                logger.info(f'step: {self.iteration}, loss: {loss}')

            self.optimizer.step()
            scheduler.step()
            self.iteration += 1

    @torch.no_grad()
    def evaluate(self, epochs: int = 1) -> float:
        self.model.eval()
        losses = []

        for _ in range(epochs):
            inputs, targets = self.batch(self.vaild_dataset)
            inputs = inputs.int()
            targets = targets.int()

            outputs = self.model(inputs)
            loss = trn.cross_entropy(outputs, targets)
            losses.append(loss)

        losses = torch.stack(losses)
        mean_loss = torch.mean(losses).item()
        logger.info(f'vaild loss: {mean_loss}')

        return mean_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=1_000, help='Total training steps')
    parser.add_argument('--save-path', type=str, default='last.pt', help='Checkpoint save path')
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.evaluate()

    trainer.train(args.epochs)
    trainer.save_checkpoint(args.save_path)


if __name__ == '__main__':
    main()

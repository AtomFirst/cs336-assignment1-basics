import os
import yaml
import argparse
import logging
from functools import partial
from typing import BinaryIO, IO

import numpy as np
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
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='config.yaml')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
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

        batch_cfg = self.config['batch']
        self.batch = partial(trnlp.get_batch, **batch_cfg, context_length=model_cfg['context_length'])

        self.iteration = 0

        checkpoint_path = self.config['checkpoint']['path']
        if checkpoint_path is not None:
            self.iteration = trnlp.load_checkpoint(checkpoint_path, self.model, self.optimizer)
    
    def save_checkpoint(self, out: str | os.PathLike | BinaryIO | IO[bytes]):
        trnlp.save_checkpoint(self.model, self.optimizer, self.iteration, out)

    def train(self, epochs: int):
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_cosine_schedule, self.iteration - 1)

        for i in range(epochs):
            self.optimizer.zero_grad()

            inputs, targets = self.batch(self.train_dataset)
            outputs = self.model(inputs)
            loss = trn.cross_entropy(outputs, targets)
            loss.backward()

            logger.info(f'loss: {loss}')

            self.optimizer.step()
            scheduler.step()
            self.iteration += 1


def main():
    trainer = Trainer()

    trainer.train(1)
    trainer.save_checkpoint('last.pt')


if __name__ == '__main__':
    main()

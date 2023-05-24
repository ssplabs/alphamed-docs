from time import time
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from alphafed import get_dataset_dir, logger
from alphafed.fed_prox import FedProxScheduler

from train_res import (get_conv_net, get_loss_fn, get_optimizer,
                       get_test_dataloader, get_train_dataloader,
                       test_process, train_an_epoch_process)


class BenchFedProx(FedProxScheduler):

    def __init__(self,
                 dataset_client_ids: List[int],
                 max_rounds: int = 0,
                 merge_epochs: int = 1,
                 mu: float = 0.01,
                 calculation_timeout: int = 300,
                 schedule_timeout: int = 30,
                 log_rounds: int = 0,
                 involve_aggregator: bool = False):
        super().__init__(max_rounds=max_rounds,
                         merge_epochs=merge_epochs,
                         mu=mu,
                         calculation_timeout=calculation_timeout,
                         schedule_timeout=schedule_timeout,
                         log_rounds=log_rounds,
                         involve_aggregator=involve_aggregator)
        self.dataset_client_ids = dataset_client_ids
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        torch.manual_seed(self.seed)

    def build_model(self) -> nn.Module:
        model = get_conv_net()
        return model.to(self.device)

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return get_optimizer(model=model)

    def build_train_dataloader(self) -> DataLoader:
        return get_train_dataloader(data_dir=get_dataset_dir(self.task_id),
                                    client_ids=self.dataset_client_ids)

    def build_test_dataloader(self) -> DataLoader:
        return get_test_dataloader(data_dir=get_dataset_dir(self.task_id),
                                   client_ids=self.dataset_client_ids)

    def train_an_epoch(self) -> None:
        train_start = time()
        train_an_epoch_process(model=self.model,
                               train_loader=self.train_loader,
                               device=self.device,
                               optimizer=self.optimizer,
                               loss_fn=get_loss_fn())
        train_end = time()
        logger.info(f'完成一轮训练，耗时 {train_end - train_start:.3f} 秒')

    def run_test(self):
        test_start = time()
        avg_loss, accuracy = test_process(model=self.model,
                                          test_loader=self.test_loader,
                                          device=self.device,
                                          loss_fn=get_loss_fn())
        test_end = time()

        logger.info(f'Test set: Average loss: {avg_loss:.4f}')
        logger.info(f'Test set: Accuracy: {accuracy} ({accuracy * 100:.2f}%)')
        logger.info(f'完成一轮测试，耗时 {test_end - test_start:.3f} 秒')

        self.tb_writer.add_scalars('test/run_time',
                                   {f'FedProx-mu:{self.mu}': test_end - test_start},
                                   self.current_round)
        self.tb_writer.add_scalars('test/average_loss',
                                   {f'FedProx-mu:{self.mu}': avg_loss},
                                   self.current_round)
        self.tb_writer.add_scalars('test/accuracy',
                                   {f'FedProx-mu:{self.mu}': accuracy},
                                   self.current_round)

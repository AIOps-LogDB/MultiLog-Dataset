#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq and self.quan and self.sem:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        elif self.seq and self.quan:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        elif self.sem:
            log['Semantics'] = torch.tensor(np.array(self.Semantics[idx]),
                                            dtype=torch.float)
        elif self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        return log, self.labels[idx]

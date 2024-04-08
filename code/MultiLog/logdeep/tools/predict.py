#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter

sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window_with_data
from sklearn.metrics import precision_score, f1_score, recall_score


def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('../data/hdfs/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        if torch.cuda.is_available():
            if "robustlog" in options['model_path']:
                self.device = torch.device("cuda:1")
            elif "loganomaly" in options['model_path']:
                self.device = torch.device("cuda:3")
            elif "dislog" in options['model_path']:
                self.device = torch.device("cuda:3")
        else:
            self.device = torch.device("cpu")
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.dataset = options['dataset']
        self.case = options['case']
        self.node_index = options['node_index']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        # model.load_state_dict(torch.load(self.model_path)['state_dict'])
        # model.eval()
        # print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate('hdfs_test_normal')
        test_abnormal_loader, test_abnormal_length = generate(
            'hdfs_test_abnormal')
        TP = 0
        FP = 0
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1])
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += test_normal_loader[line]
                        break
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(
                        -1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1])
                    predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                    if label not in predicted:
                        TP += test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def get_by_blocks(self, instances, block_list):
        result = []
        labels = []
        for instance in instances:
            if instance.id in block_list:
                result.append([int(i) for i in instance.sequence])
                if instance.label == 'Anomalous':
                    labels.append(1)
                else:
                    labels.append(0)
        return result, labels

    def predict_supervised(self, result_file, time_list, time_index2block, instances):
        self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(self.model_path)['state_dict'])
        # model.eval()
        # print('model_path: {}'.format(self.model_path))
        TP, FP, FN, TN = 0, 0, 0, 0
        for time_index in range(len(time_list)):
            if time_index in time_index2block:
                logs, labels = self.get_by_blocks(instances, time_index2block[time_index])
                test_logs, test_labels = session_window_with_data(self.data_dir, dataset=self.dataset,
                                                                  case=self.case,
                                                                  node_index=self.node_index, data=logs, labels=labels,
                                                                  sequentials=self.sequentials,
                                                                  quantitatives=self.quantitatives,
                                                                  semantics=self.semantics)
                test_dataset = log_dataset(logs=test_logs,
                                           labels=test_labels,
                                           seq=self.sequentials,
                                           quan=self.quantitatives,
                                           sem=self.semantics)
                self.test_loader = DataLoader(test_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=32)
                tbar = tqdm(self.test_loader, desc="\r")
                result = []
                for i, (log, label) in enumerate(tbar):
                    features = []
                    for value in log.values():
                        features.append(value.clone().to(self.device))
                    output = self.model(features=features)
                    output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
                    result.extend(output)
                    # predicted = torch.argmax(output, dim=1).cpu().numpy()
                    predicted = (output < 0.2).astype(int)
                    label = np.array([y.cpu() for y in label])
                    TP += ((predicted == 1) * (label == 1)).sum()
                    FP += ((predicted == 1) * (label == 0)).sum()
                    FN += ((predicted == 0) * (label == 1)).sum()
                    TN += ((predicted == 0) * (label == 0)).sum()
                result_file.write(str(time_index) + ":" + str(result) + "\n")
                result_file.flush()
            else:
                result_file.write(str(time_index) + ":" + str([]) + "\n")
                result_file.flush()

        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        return P, R, F1

    def predict_supervised_multiclass(self, result_file, time_list, time_index2block, instances, truth_label_list):
        self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(self.model_path)['state_dict'])
        # model.eval()
        # print('model_path: {}'.format(self.model_path))
        label_list = []
        pred_label_list = []
        for time_index in range(len(time_list)):
            if time_index in time_index2block:
                logs, labels = self.get_by_blocks(instances, time_index2block[time_index])
                label = truth_label_list[time_index]
                if len(logs) <= 0:
                    label_list.append(label)
                    pred_label_list.append(0)
                    result_file.write(
                        str(time_index) + ":" + str(time_list[time_index]) + ":" + str(0) + "\n")
                    continue
                test_logs, test_labels = session_window_with_data(self.data_dir, dataset=self.dataset,
                                                                  case=self.case,
                                                                  node_index=self.node_index, data=logs, labels=labels,
                                                                  sequentials=self.sequentials,
                                                                  quantitatives=self.quantitatives,
                                                                  semantics=self.semantics, multi_class=True)
                test_dataset = log_dataset(logs=test_logs,
                                           labels=test_labels,
                                           seq=self.sequentials,
                                           quan=self.quantitatives,
                                           sem=self.semantics)
                self.test_loader = DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=32)
                tbar = tqdm(self.test_loader, desc="\r")
                result = []
                pred_labels = []
                for i, (log, label) in enumerate(tbar):
                    features = []
                    for value in log.values():
                        features.append(value.clone().to(self.device))
                    print("featuresLen=" + str(len(features)))
                    output = self.model(features=features)
                    result.extend(output)
                    print(output)
                    predicted = torch.argsort(output, 1)[0][-1:].cpu()
                    print(predicted)
                    pred_labels.append(predicted)
                label = truth_label_list[time_index]
                # 使用 np.unique 函数找到数组中的唯一值及其出现次数
                unique_values, counts = np.unique(pred_labels, return_counts=True)
                # 找到出现次数最多的索引
                most_common_index = np.argmax(counts)
                # 获取众数
                predicted_label = unique_values[most_common_index]
                pred_label_list.append(predicted_label)
                label_list.append(label)
                print("pred_labels=" + str(pred_labels))
                print("predicted_label=" + str(predicted_label))
                print("label=" + str(label))
                result_file.write(str(time_index) + ":" + str(result) + "\n")
                result_file.flush()
            else:
                result_file.write(str(time_index) + ":" + str([]) + "\n")
                result_file.flush()

        precision = precision_score(label_list, pred_label_list, average="macro")
        recall = recall_score(label_list, pred_label_list, average="macro")
        f1 = f1_score(label_list, pred_label_list, average="macro")
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(precision, recall, f1))
        return precision, recall, f1

    def predict_supervised_by_time(self, time_list, time_index2block, instances, time_index2label, result_file):
        self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(self.model_path)['state_dict'])
        # model.eval()
        # print('model_path: {}'.format(self.model_path))
        TP, FP, FN, TN = 0, 0, 0, 0
        for time_index in range(len(time_list)):
            if time_index in time_index2block:
                logs, labels = self.get_by_blocks(instances, time_index2block[time_index])
                label = time_index2label[time_index]
                if len(logs) <= 0:
                    if label == 0:
                        TP += 1
                    else:
                        FN += 1
                    result_file.write(
                        str(time_index) + ":" + str(time_list[time_index]) + ":" + str(0) + "\n")
                    continue
                test_logs, test_labels = session_window_with_data(self.data_dir, dataset=self.dataset,
                                                                  case=self.case,
                                                                  node_index=self.node_index, data=logs, labels=labels,
                                                                  sequentials=self.sequentials,
                                                                  quantitatives=self.quantitatives,
                                                                  semantics=self.semantics)
                test_dataset = log_dataset(logs=test_logs,
                                           labels=test_labels,
                                           seq=self.sequentials,
                                           quan=self.quantitatives,
                                           sem=self.semantics)
                self.test_loader = DataLoader(test_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=32)
                tbar = tqdm(self.test_loader, desc="\r")
                pred_labels = []
                for i, (log, label) in enumerate(tbar):
                    features = []
                    for value in log.values():
                        features.append(value.clone().to(self.device))
                    output = self.model(features=features)
                    output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
                    predicted = np.max((output < 0.2).astype(int))
                    pred_labels.append(predicted)
                label = time_index2label[time_index]
                predicted_label = np.max(pred_labels)
                result_file.write(
                    str(time_index) + ":" + str(time_list[time_index]) + ":" + str(predicted_label) + "\n")
                if predicted_label == 1:
                    if label == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if label == 1:
                        FN += 1
                    else:
                        TN += 1

        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        return P, R, F1

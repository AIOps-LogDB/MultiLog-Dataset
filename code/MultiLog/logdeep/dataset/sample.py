import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


def session_window(data_dir, datatype, sample_ratio=1, dataset="hdfs", case="cpu_continue", node_index="1",
                   sequentials=False, quantitatives=False, semantics=False, multi_class=False):
    if case in ["multi_anomaly_single_node", "multi_anomaly_single_node_reduced"]:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-3_st-0.3/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-3_st-0.3/templates.vec',
                'r', encoding='utf-8')
    elif case in ["multi_anomaly_multi_node"]:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-3_st-0.2/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-3_st-0.2/templates.vec',
                'r', encoding='utf-8')
    else:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-4_st-0.4/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-4_st-0.4/templates.vec',
                'r', encoding='utf-8')
    event2semantic_vec = {}
    for line in reader.readlines():
        token = line.split()
        template_id = int(token[0])
        embed = np.asarray(token[1:], dtype=np.float64)
        event2semantic_vec[template_id] = embed

    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    data = []
    labels = []

    max_normal_num = 1000
    curr_normal_num = 0
    if datatype == 'train':
        if multi_class:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}_multi/inputs/IBM/train')
        else:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}/inputs/IBM/train')
        for line in reader.readlines():
            if len(line.strip()) > 0:
                if "," in line:
                    if multi_class:
                        label = int(line.split(",")[-1])
                        if label == 0 and curr_normal_num > max_normal_num:
                            data.pop()
                            continue
                        labels.append(label)
                        if label == 0:
                            curr_normal_num += 1
                    else:
                        if "Anomalous" in line:
                            labels.append(1)
                        else:
                            labels.append(0)
                else:
                    data.append([int(i) for i in line.split(" ")])
        reader.close()
    elif datatype == 'val':
        if multi_class:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}_multi/inputs/IBM/dev')
        else:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}/inputs/IBM/dev')
        for line in reader.readlines():
            if len(line.strip()) > 0:
                if "," in line:
                    if multi_class:
                        labels.append(int(line.split(",")[-1]))
                    else:
                        if "Anomalous" in line:
                            labels.append(1)
                        else:
                            labels.append(0)
                else:
                    data.append([int(i) for i in line.split(" ")])
        reader.close()
    elif datatype == 'test':
        if multi_class:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}_multi/inputs/IBM/test')
        else:
            reader = open(data_dir + f'{dataset}/{case}/{node_index}/inputs/IBM/test')
        for line in reader.readlines():
            if len(line.strip()) > 0:
                if "," in line:
                    if multi_class:
                        labels.append(int(line.split(",")[-1]))
                    else:
                        if "Anomalous" in line:
                            labels.append(1)
                        else:
                            labels.append(0)
                else:
                    data.append([int(i) for i in line.split(" ")])
        reader.close()

    result_labels = []
    if sequentials and quantitatives and semantics:
        for i in tqdm(range(len(data))):
            ori_seq = data[i]
            Sequential_pattern = trp(ori_seq, 500)
            Quantitative_pattern = [0] * (len(event2semantic_vec) + 1)
            Semantic_pattern = [0] * 300
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            for event in ori_seq:
                Semantic_pattern = list(np.add(Semantic_pattern, event2semantic_vec[event]))

            Semantic_pattern = np.array(Semantic_pattern)[:, np.newaxis]
            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_labels.append(labels[i])
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
    elif sequentials and quantitatives:
        for i in tqdm(range(len(data))):
            ori_seq = data[i]
            Sequential_pattern = trp(ori_seq, 500)
            Quantitative_pattern = [0] * (len(event2semantic_vec) + 1)
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_labels.append(labels[i])
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
    elif semantics:
        for i in tqdm(range(len(data))):
            Semantic_pattern = []
            for event in data[i]:
                if len(Semantic_pattern) >= 100:
                    result_labels.append(labels[i])
                    result_logs['Semantics'].append(Semantic_pattern)
                    Semantic_pattern = []
                if event in event2semantic_vec:
                    Semantic_pattern.append(event2semantic_vec[event])
                else:
                    Semantic_pattern.append([-1] * 300)
            while len(Semantic_pattern) < 100:
                Semantic_pattern.append([-1] * 300)

            result_labels.append(labels[i])
            result_logs['Semantics'].append(Semantic_pattern)

    if sample_ratio != 1:
        result_logs, result_labels = down_sample(result_logs, result_labels, sample_ratio)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, result_labels


def session_window_with_data(data_dir, dataset="hdfs", case="cpu_continue", node_index="1", data=None, labels=None,
                             sequentials=False, quantitatives=False, semantics=False, multi_class=False):
    if case in ["multi_anomaly_single_node", "multi_anomaly_single_node_reduced"]:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-3_st-0.3/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-3_st-0.3/templates.vec',
                'r', encoding='utf-8')
    elif case in ["multi_anomaly_multi_node"]:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-3_st-0.2/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-3_st-0.2/templates.vec',
                'r', encoding='utf-8')
    else:
        if multi_class:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}_multi/persistences/ibm_drain_depth-4_st-0.4/templates.vec',
                'r', encoding='utf-8')
        else:
            reader = open(
                data_dir + f'{dataset}/{case}/{node_index}/persistences/ibm_drain_depth-4_st-0.4/templates.vec',
                'r', encoding='utf-8')
    event2semantic_vec = {}
    for line in reader.readlines():
        token = line.split()
        template_id = int(token[0])
        embed = np.asarray(token[1:], dtype=np.float64)
        event2semantic_vec[template_id] = embed

    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []

    result_labels = []
    if sequentials and quantitatives and semantics:
        for i in tqdm(range(len(data))):
            ori_seq = data[i]
            Sequential_pattern = trp(ori_seq, 500)
            Quantitative_pattern = [0] * (len(event2semantic_vec) + 1)
            Semantic_pattern = [0] * 300
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            for event in ori_seq:
                Semantic_pattern = list(np.add(Semantic_pattern, event2semantic_vec[event]))

            Semantic_pattern = np.array(Semantic_pattern)[:, np.newaxis]
            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_labels.append(labels[i])
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
    elif sequentials and quantitatives:
        for i in tqdm(range(len(data))):
            ori_seq = data[i]
            Sequential_pattern = trp(ori_seq, 500)
            Quantitative_pattern = [0] * (len(event2semantic_vec) + 1)
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_labels.append(labels[i])
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
    elif semantics:
        for i in tqdm(range(len(data))):
            Semantic_pattern = []
            for event in data[i]:
                if len(Semantic_pattern) >= 100:
                    result_labels.append(labels[i])
                    result_logs['Semantics'].append(Semantic_pattern)
                    Semantic_pattern = []
                if event in event2semantic_vec:
                    Semantic_pattern.append(event2semantic_vec[event])
                else:
                    Semantic_pattern.append([-1] * 300)
            while len(Semantic_pattern) < 100:
                Semantic_pattern.append([-1] * 300)

            result_labels.append(labels[i])
            result_logs['Semantics'].append(Semantic_pattern)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, result_labels

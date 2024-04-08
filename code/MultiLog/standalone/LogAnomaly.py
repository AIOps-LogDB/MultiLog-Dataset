#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import shutil
import sys

sys.path.append('../')

from CONSTANTS import *
from logdeep.models.lstm import loganomaly
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.Preprocess import Preprocessor
from datetime import datetime, timedelta

seed_everything(seed=1234)


def train_and_test_time(dataset, CASE, node_list, time_window_size):
    result_file = open(f'result_time_predict_LogAnomaly_{CASE}.log', "w")

    log_start_time = 999999999999999
    log_end_time = 0
    for node in node_list:
        in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/label" + node + '.log')
        for line in open(in_file, "r"):
            time_str = line.split("[")[0][:-1].replace("- ", "")
            timestamp = (datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f") + timedelta(hours=8)).timestamp()
            log_start_time = min(timestamp, log_start_time)
            log_end_time = max(timestamp, log_end_time)

    inject_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/inject.log")
    inject_anomaly_types = []
    start_time = []
    end_time = []
    for line in open(inject_file):
        if "start inject" in line:
            anomaly = line.split("start inject ")[1].strip()
            if "none" not in anomaly:
                inject_anomaly_types.append(1)
            else:
                inject_anomaly_types.append(0)
        else:
            if "Recover" not in line and "inject" not in line and ("169" in line or "170" in line) and "." in line:
                time_float = float(line)
                if len(end_time) < len(start_time):
                    end_time.append(time_float)
                else:
                    start_time.append(time_float)

    time_list = []
    curr_start_time = log_start_time
    while log_end_time > curr_start_time:
        time_list.append([curr_start_time, curr_start_time + time_window_size])
        curr_start_time = curr_start_time + time_window_size

    truth_label_list = {}
    for i in range(len(time_list)):
        truth_label_list[i] = 0
        for j in range(len(start_time)):
            if not (start_time[j] > time_list[i][1] or end_time[j] < time_list[i][0]):
                if inject_anomaly_types[j] == 1:
                    truth_label_list[i] = 1
                    break

    for time_index in range(len(time_list)):
        result_file.write(
            str(time_index) + ":" + str(time_list[time_index]) + ":" + str(truth_label_list[time_index]) + "\n")
    result_file.flush()

    parser = 'IBM'

    result = []
    for node in node_list:
        result_file.write("===node:" + node + "===" + "\n")
        # Training, Validating and Testing instances.
        template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
        processor = Preprocessor()

        train, dev, instances, time_index2block = processor.process(dataset=dataset, case=CASE,
                                                                    node_index=node, parsing=parser,
                                                                    cut_func=cut_by_613,
                                                                    template_encoding=template_encoder.present,
                                                                    time_list=time_list)

        # Config Parameters

        options = dict()
        options['data_dir'] = 'datasets/'
        options['sample'] = "session_window"
        options['window_size'] = 100

        # Features
        options['sequentials'] = True
        options['quantitatives'] = True
        options['semantics'] = False
        options['feature_num'] = sum(
            [options['sequentials'], options['quantitatives'], options['semantics']])

        # Model
        options['input_size'] = 1
        options['hidden_size'] = 64
        options['num_layers'] = 2
        options['num_classes'] = 2

        # Train
        options['batch_size'] = 256
        options['accumulation_step'] = 1

        options['optimizer'] = 'adam'
        options['lr'] = 0.001
        options['lr_step'] = (300, 350)
        options['lr_decay_ratio'] = 0.1
        if dataset == "BGL":
            options['max_epoch'] = 60
        elif dataset == "HDFS" or dataset == "Thunderbird":
            options['max_epoch'] = 40
        else:
            options['max_epoch'] = 400

        options['resume_path'] = None
        options['model_name'] = "loganomaly"
        options['save_dir'] = "../result/loganomaly/"

        # Predict
        options['model_path'] = f"../result/loganomaly/loganomaly_last_{CASE}_{node}.pth"
        options['num_candidates'] = -1

        options['dataset'] = dataset
        options['case'] = CASE
        options['node_index'] = node

        seed_everything(seed=1234)

        Model = loganomaly(input_size=options['input_size'],
                           hidden_size=options['hidden_size'],
                           num_layers=options['num_layers'],
                           num_keys=options['num_classes'])
        trainer = Trainer(Model, options)
        trainer.start_train()
        predicter = Predicter(Model, options)
        result.append(
            predicter.predict_supervised_by_time(time_list, time_index2block, instances, truth_label_list, result_file))
    result_file.close()
    return result


def train_and_test_time_single_node(dataset, CASE, NODE, time_window_size):
    result_file = open(f'result_time_predict_LogAnomaly_{CASE}_{NODE}.log', "w")

    log_start_time = 999999999999999
    log_end_time = 0
    for node in ["1", "2", "3", "4"]:
        in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/label" + node + '.log')
        for line in open(in_file, "r"):
            time_str = line.split("[")[0][:-1].replace("- ", "")
            timestamp = (datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f") + timedelta(hours=8)).timestamp()
            log_start_time = min(timestamp, log_start_time)
            log_end_time = max(timestamp, log_end_time)

    inject_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/inject.log")
    inject_anomaly_types = []
    start_time = []
    end_time = []
    for line in open(inject_file):
        if "start inject" in line:
            anomaly = line.split("start inject ")[1].strip()
            if "none" not in anomaly:
                inject_anomaly_types.append(1)
            else:
                inject_anomaly_types.append(0)
        else:
            if "Recover" not in line and "inject" not in line and ("169" in line or "170" in line) and "." in line:
                time_float = float(line)
                if len(end_time) < len(start_time):
                    end_time.append(time_float)
                else:
                    start_time.append(time_float)

    time_list = []
    curr_start_time = log_start_time
    while log_end_time > curr_start_time:
        time_list.append([curr_start_time, curr_start_time + time_window_size])
        curr_start_time = curr_start_time + time_window_size

    truth_label_list = {}
    for i in range(len(time_list)):
        truth_label_list[i] = 0
        for j in range(len(start_time)):
            if not (start_time[j] > time_list[i][1] or end_time[j] < time_list[i][0]):
                if inject_anomaly_types[j] == 1:
                    truth_label_list[i] = 1
                    break

    for time_index in range(len(time_list)):
        result_file.write(
            str(time_index) + ":" + str(time_list[time_index]) + ":" + str(truth_label_list[time_index]) + "\n")
    result_file.flush()

    result_file.write(f"===node:{NODE}===\n")

    parser = 'IBM'

    # Training, Validating and Testing instances.
    template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor = Preprocessor()

    train, dev, instances, time_index2block = processor.process(dataset=dataset, case=CASE,
                                                                node_index=NODE, parsing=parser,
                                                                cut_func=cut_by_613,
                                                                template_encoding=template_encoder.present,
                                                                time_list=time_list)

    # Config Parameters

    options = dict()
    options['data_dir'] = 'datasets/'
    options['sample'] = "session_window"
    options['window_size'] = 100

    # Features
    options['sequentials'] = True
    options['quantitatives'] = True
    options['semantics'] = False
    options['feature_num'] = sum(
        [options['sequentials'], options['quantitatives'], options['semantics']])

    # Model
    options['input_size'] = 1
    options['hidden_size'] = 64
    options['num_layers'] = 2
    options['num_classes'] = 2

    # Train
    options['batch_size'] = 256
    options['accumulation_step'] = 1

    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    options['lr_step'] = (300, 350)
    options['lr_decay_ratio'] = 0.1
    if dataset == "BGL":
        options['max_epoch'] = 60
    elif dataset == "HDFS" or dataset == "Thunderbird":
        options['max_epoch'] = 40
    else:
        options['max_epoch'] = 400

    options['resume_path'] = None
    options['model_name'] = "loganomaly"
    options['save_dir'] = "../result/loganomaly/"

    # Predict
    options['model_path'] = f"../result/loganomaly/loganomaly_last_{CASE}_{NODE}.pth"
    options['num_candidates'] = -1

    options['dataset'] = dataset
    options['case'] = CASE
    options['node_index'] = NODE

    seed_everything(seed=1234)

    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()
    predicter = Predicter(Model, options)
    result = predicter.predict_supervised_by_time(time_list, time_index2block, instances, truth_label_list, result_file)
    result_file.close()
    return result


def get_reduced_result_by_time(dataset, CASE, NODE, time_list, time_index2label):
    parser = 'IBM'

    # Training, Validating and Testing instances.
    template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor = Preprocessor()

    train, dev, instances, time_index2block = processor.process(dataset=dataset, case=CASE,
                                                                node_index=NODE, parsing=parser,
                                                                cut_func=cut_by_613,
                                                                template_encoding=template_encoder.present,
                                                                time_list=time_list)

    # Config Parameters

    options = dict()
    options['data_dir'] = 'datasets/'
    options['sample'] = "session_window"
    options['window_size'] = 100

    # Features
    options['sequentials'] = True
    options['quantitatives'] = True
    options['semantics'] = False
    options['feature_num'] = sum(
        [options['sequentials'], options['quantitatives'], options['semantics']])

    # Model
    options['input_size'] = 1
    options['hidden_size'] = 64
    options['num_layers'] = 2
    options['num_classes'] = 2

    # Train
    options['batch_size'] = 256
    options['accumulation_step'] = 1

    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    options['lr_step'] = (300, 350)
    options['lr_decay_ratio'] = 0.1
    if dataset == "BGL":
        options['max_epoch'] = 60
    elif dataset == "HDFS" or dataset == "Thunderbird":
        options['max_epoch'] = 40
    else:
        options['max_epoch'] = 400

    options['resume_path'] = None
    options['model_name'] = "loganomaly"
    options['save_dir'] = "../result/loganomaly/"

    # Predict
    options['model_path'] = f"../result/loganomaly/loganomaly_last_{CASE}_{NODE}.pth"
    options['num_candidates'] = -1

    options['dataset'] = dataset
    options['case'] = CASE
    options['node_index'] = NODE

    seed_everything(seed=1234)

    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()
    predicter = Predicter(Model, options)
    result = predicter.predict_supervised_by_time(time_list, time_index2block, instances, time_index2label)
    return result


f1_down_ratio = 0.02


def clean_more_log(dataset, CASE, NODE, time_window_size):
    result_file = open(f'result_time_predict_LogAnomaly_clean_{CASE}_{NODE}.log', "w")

    log_start_time = 999999999999999
    log_end_time = 0
    for node in ["1", "2", "3", "4"]:
        in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/label" + node + '.log')
        for line in open(in_file, "r"):
            time_str = line.split("[")[0][:-1].replace("- ", "")
            timestamp = (datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f") + timedelta(hours=8)).timestamp()
            log_start_time = min(timestamp, log_start_time)
            log_end_time = max(timestamp, log_end_time)

    inject_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/inject.log")
    inject_anomaly_types = []
    start_time = []
    end_time = []
    for line in open(inject_file):
        if "start inject" in line:
            anomaly = line.split("start inject ")[1].strip()
            if "none" not in anomaly:
                inject_anomaly_types.append(1)
            else:
                inject_anomaly_types.append(0)
        else:
            if "Recover" not in line and "inject" not in line and ("169" in line or "170" in line) and "." in line:
                time_float = float(line)
                if len(end_time) < len(start_time):
                    end_time.append(time_float)
                else:
                    start_time.append(time_float)

    time_list = []
    curr_start_time = log_start_time
    while log_end_time > curr_start_time:
        time_list.append([curr_start_time, curr_start_time + time_window_size])
        curr_start_time = curr_start_time + time_window_size

    truth_label_list = {}
    for i in range(len(time_list)):
        truth_label_list[i] = 0
        for j in range(len(start_time)):
            if not (start_time[j] > time_list[i][1] or end_time[j] < time_list[i][0]):
                if inject_anomaly_types[j] == 1:
                    truth_label_list[i] = 1
                    break

    for time_index in range(len(time_list)):
        result_file.write(
            str(time_index) + ":" + str(time_list[time_index]) + ":" + str(truth_label_list[time_index]) + "\n")
    result_file.flush()

    all_remain_class = set()
    in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "/label" + NODE + '.log')
    reduced_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced/label" + NODE + '.log')
    if os.path.exists(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced"))
    os.makedirs(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced/" + NODE))

    reduced_w = open(reduced_file, "w")
    for line in open(in_file, "r"):
        if not line.startswith("-"):
            event_id = line.split(" ")[4]
            if len(event_id) == 0:
                event_id = line.split(" ")[5]
        else:
            event_id = line.split(" ")[5]
            if len(event_id) == 0:
                event_id = line.split(" ")[6]
        if len(event_id.strip()) == 0:
            print(line)
        all_remain_class.add(event_id.split(":")[0])
        reduced_w.write(line)
    reduced_w.flush()
    reduced_w.close()

    result_file.write(str(all_remain_class) + "\n")
    result_file.flush()
    max_precision, max_recall, max_f1 = get_reduced_result_by_time(dataset, CASE, NODE, time_list, truth_label_list)
    result_file.write(str([max_precision, max_recall, max_f1]) + "\n")

    all_remain_class = list(all_remain_class)
    total_events = len(all_remain_class)
    model_anti_class = []
    model_important_class = []
    while True:
        result_file.write(str(len(all_remain_class)) + "\n")
        result_file.flush()
        can_find_next = False
        for remain_class in all_remain_class[:]:
            if remain_class in model_important_class:
                continue
            all_remain_class.remove(remain_class)

            if os.path.exists(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced")):
                shutil.rmtree(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced"))
            os.makedirs(os.path.join(PROJECT_ROOT, 'datasets/' + dataset + "/" + CASE + "_reduced/" + NODE))
            reduced_w = open(reduced_file, "w")
            for line in open(in_file, "r"):
                if not line.startswith("-"):
                    event_id = line.split(" ")[4]
                    if len(event_id) == 0:
                        event_id = line.split(" ")[5]
                else:
                    event_id = line.split(" ")[5]
                    if len(event_id) == 0:
                        event_id = line.split(" ")[6]
                if event_id.split(":")[0] != remain_class:
                    reduced_w.write(line)
            reduced_w.flush()
            reduced_w.close()

            precision, recall, f1 = get_reduced_result_by_time(dataset, CASE, NODE, time_list, truth_label_list)
            result_file.write(str([precision, recall, f1]) + "\n")
            result_file.flush()
            if f1 < (1 - f1_down_ratio) * max_f1:
                # important template and add it back
                all_remain_class.append(remain_class)
                model_important_class.append(remain_class)
            else:
                if max_f1 < f1:
                    max_precision = precision
                    max_recall = recall
                    max_f1 = f1
                    model_anti_class.append(remain_class)
                can_find_next = True
                break
        if not can_find_next:
            break

    result_file.write("template:" + str(all_remain_class) + "\n")
    result_file.write("total_events:" + str(total_events) + "\n")
    result_file.write("reduced_events:" + str(len(all_remain_class)) + "\n")
    result_file.write("max_precision:" + str(max_precision) + "\n")
    result_file.write("max_recall:" + str(max_recall) + "\n")
    result_file.write("max_f1:" + str(max_f1) + "\n")
    result_file.write("anti_template:" + str(model_anti_class) + "\n")
    result_file.write("important_template:" + str(model_important_class) + "\n")
    result_file.flush()
    result_file.close()

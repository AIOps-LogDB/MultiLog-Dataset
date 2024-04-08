#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

sys.path.append('../')

from CONSTANTS import *
from logdeep.models.lstm import robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.Preprocess import Preprocessor
from datetime import datetime, timedelta


def train_and_test_time(dataset, CASE, node_list, time_window_size):
    result_file = open(f'result_time_predict_RobustLog_{CASE}.log', "w")

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
        options['sequentials'] = False
        options['quantitatives'] = False
        options['semantics'] = True
        options['feature_num'] = sum(
            [options['sequentials'], options['quantitatives'], options['semantics']])

        # Model
        options['input_size'] = 300
        options['hidden_size'] = 128
        options['num_layers'] = 2
        options['num_classes'] = 2

        # Train
        options['batch_size'] = 256
        options['accumulation_step'] = 1

        options['optimizer'] = 'adam'
        options['lr'] = 0.001
        if dataset == "BGL":
            options['max_epoch'] = 60
        elif dataset == "HDFS" or dataset == "Thunderbird":
            options['max_epoch'] = 40
        else:
            options['max_epoch'] = 200
        options['lr_step'] = (40, 50)
        options['lr_decay_ratio'] = 0.1

        options['resume_path'] = None
        options['model_name'] = "robustlog"
        options['save_dir'] = "../result/robustlog/"

        # Predict
        options['model_path'] = f"../result/robustlog/robustlog_last_{CASE}_{node}.pth"
        options['num_candidates'] = -1

        options['dataset'] = dataset
        options['case'] = CASE
        options['node_index'] = node

        seed_everything(seed=1234)

        Model = robustlog(input_size=options['input_size'],
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
    result_file = open(f'result_time_predict_RobustLog_{CASE}_{NODE}.log', "w")

    log_start_time = 999999999999999
    log_end_time = 0
    for node in ["1", "2", "3", "4", "5", "6"]:
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
    options['sequentials'] = False
    options['quantitatives'] = False
    options['semantics'] = True
    options['feature_num'] = sum(
        [options['sequentials'], options['quantitatives'], options['semantics']])

    # Model
    options['input_size'] = 300
    options['hidden_size'] = 128
    options['num_layers'] = 2
    options['num_classes'] = 2

    # Train
    options['batch_size'] = 256
    options['accumulation_step'] = 1

    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    if dataset == "BGL":
        options['max_epoch'] = 60
    elif dataset == "HDFS" or dataset == "Thunderbird":
        options['max_epoch'] = 40
    else:
        options['max_epoch'] = 200
    options['lr_step'] = (40, 50)
    options['lr_decay_ratio'] = 0.1

    options['resume_path'] = None
    options['model_name'] = "robustlog"
    options['save_dir'] = "../result/robustlog/"

    # Predict
    options['model_path'] = f"../result/robustlog/robustlog_last_{CASE}_{NODE}.pth"
    options['num_candidates'] = -1

    options['dataset'] = dataset
    options['case'] = CASE
    options['node_index'] = NODE

    seed_everything(seed=1234)

    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()
    predicter = Predicter(Model, options)
    result = predicter.predict_supervised_by_time(time_list, time_index2block, instances, truth_label_list, result_file)
    result_file.close()
    return result

import os
import numpy as np


def get_data_and_time_label(file):
    data = []
    time_label = []
    for line in open(file, "r"):
        if "===node:" in line:
            break
        else:
            time_label.append(int(line.split(":")[-1]))
    node_dict = {1: [0] * len(time_label), 2: [0] * len(time_label), 3: [0] * len(time_label),
                 4: [0] * len(time_label)}
    curr_node = -1
    for line in open(file, "r"):
        if "===node:" in line:
            curr_node = int(line.split(":")[1].replace("=", ""))
        else:
            if curr_node != -1:
                time_index = int(line.split(":")[0])
                label = int(line.split(":")[-1])
                node_dict[curr_node][time_index] = label

    while len(node_dict) < 4:
        node_dict[len(node_dict) + 1] = []

    for time_index in range(len(time_label)):
        node_prob_list = []
        for node in node_dict.keys():
            if len(node_dict[node]) > time_index:
                node_prob_list.append(node_dict[node][time_index])
            else:
                node_prob_list.append(0)
        data.append(node_prob_list)
    return data, time_label

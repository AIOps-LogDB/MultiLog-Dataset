import sys

from sklearn.metrics import precision_score, recall_score, f1_score

from dis_common import get_data_and_time_label

case_list = ["multi_anomaly_single_node", "multi_anomaly_single_node_reduced"]
node_num = 6

def calculate_label(time_segments):
    return max(time_segments)


MODEL = "LogAnomaly"

for case in case_list:
    data, time_label = get_data_and_time_label(f"result_time_predict_{MODEL}_{case}.log")

    predicted_labels = [calculate_label(time_segment_data) for time_segment_data in data]

    precision = precision_score(time_label, predicted_labels)
    recall = recall_score(time_label, predicted_labels)
    f1 = f1_score(time_label, predicted_labels)

    print(f"{case}:[{precision},{recall},{f1}]")


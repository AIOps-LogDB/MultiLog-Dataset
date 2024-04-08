import copy
import sys
import traceback

sys.path.extend([".", ".."])
from CONSTANTS import *
from loglizer import dataloader
from loglizer.models import *
from loglizer import preprocessing
from sklearnex import patch_sklearn

patch_sklearn()


def get_model_res(_model, x_tr, y_train, x_te, y_test):
    model = None
    feature_extractor = None
    if _model == 'PCA':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                  normalization='zero-mean')
        model = PCA()
        model.fit(x_train)

    elif _model == 'InvariantsMiner':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr)
        model = InvariantsMiner(epsilon=0.5)
        model.fit(x_train)

    elif _model == 'LogClustering':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
        model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
        model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

    elif _model == 'IsolationForest':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr)
        model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03,
                                n_jobs=4)
        model.fit(x_train)

    elif _model == 'LR':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
        model = LR()
        model.fit(x_train, y_train)

    elif _model == 'SVM':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
        model = SVM()
        model.fit(x_train, y_train)

    elif _model == 'DecisionTree':
        feature_extractor = preprocessing.FeatureExtractor()
        x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
        model = DecisionTree()
        model.fit(x_train, y_train)

    x_test = feature_extractor.transform(x_te)
    precision, recall, f1 = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    if np.all(y_pred == 1) or np.all(y_pred == 0):
        print("Cannot Classify")
        return 0, 0, 0

    return precision, recall, f1, model, feature_extractor


def train_and_test_time(_model, CASE, node_list):
    window_size = 100
    train_ratio = 0.5
    time_index_label_file = os.path.join(PROJECT_ROOT, f'datasets/IoTDB/{CASE}/time_index_label.txt')
    time_index2label = {}
    for line in open(time_index_label_file, "r"):
        time_index2label[int(line.split(":")[0])] = int(line.split(":")[1])
    for node_index in node_list:
        (x_tr, y_train), (x_te, y_test) = dataloader.load_npz(
            os.path.join(PROJECT_ROOT,
                         f'datasets/IoTDB/{CASE}/data{str(node_index)}.npz'),
            train_ratio=train_ratio,
            split_type='uniform')
        try:
            precision, recall, f1, model, feature_extractor = get_model_res(_model, x_tr, y_train, x_te, y_test)
            TP, TN, FP, FN = 0, 0, 0, 0
            time_index_node_file = os.path.join(PROJECT_ROOT,
                                                f'datasets/IoTDB/{CASE}/time_index_node{str(node_index)}.txt')
            time_index2blk = {}
            curr_blk_id = 0
            splitted_event = []
            for line in open(time_index_node_file, "r"):
                time_index = int(line.split(":")[0])
                if "," in line:
                    event_list_str = line.split(":")[1].replace("[", "").replace("]", "").replace("\n", "")
                    event_list = [e.replace("'", "").replace(" ", "") for e in event_list_str.split(",")]
                else:
                    event_list = []
                blk_id_list = []
                curr_event_list = []
                for event in event_list:
                    curr_event_list.append(event)
                    if len(curr_event_list) >= window_size:
                        splitted_event.append(curr_event_list)
                        curr_event_list = []
                        blk_id_list.append(curr_blk_id)
                        curr_blk_id += 1
                if len(curr_event_list) > 0:
                    splitted_event.append(curr_event_list)
                    blk_id_list.append(curr_blk_id)
                    curr_blk_id += 1
                time_index2blk[time_index] = blk_id_list

            x_data_array = np.empty(len(splitted_event), dtype=list)
            for i in range(len(splitted_event)):
                x_data_array[i] = splitted_event[i]
            all_test = feature_extractor.transform(x_data_array)
            result = model.predict(all_test)

            for time_index, blk_id_list in time_index2blk.items():
                if len(blk_id_list) <= 0:
                    label = 0
                else:
                    time_result = result[blk_id_list]
                    label = np.max(time_result)
                if label == 1:
                    if time_index2label[time_index] == 0:
                        FP += 1
                    else:
                        TP += 1
                else:
                    if time_index2label[time_index] == 0:
                        TN += 1
                    else:
                        FN += 1
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            print('TP:', TP, 'TN:', TN, 'FP:', FP, 'FN:', FN, "result:", [precision, recall, f1])
        except Exception as e:
            traceback.print_exc()


f1_down_ratio = 0.02


def clean_more_log(_model, CASE, node_index):
    data_file = f'../datasets/IoTDB/{CASE}/data{str(node_index)}.npz'
    data = np.load(data_file, allow_pickle=True)
    x_data = data['x_data']
    total_line = 0
    all_templates = set()
    for i in range(len(x_data)):
        all_templates.update(x_data[i])
        total_line += len(x_data[i])
    all_templates = sorted(list(all_templates))
    total_events = len(all_templates)

    train_ratio = 0.7
    (original_x_tr, y_train), (original_x_te, y_test) = dataloader.load_npz(
        f'../datasets/IoTDB/{CASE}/data{str(node_index)}.npz',
        train_ratio=train_ratio,
        split_type='uniform')
    max_precision, max_recall, max_f1 = get_model_res(_model, original_x_tr, y_train, original_x_te, y_test)

    log_writer = open(f"../datasets/LogCleanerX.log", "w")
    log_writer.write(str([max_precision, max_recall, max_f1]) + "\n")
    model_anti_template = []
    model_important_template = []
    while True:
        log_writer.write(str(len(all_templates)) + "\n")
        log_writer.flush()
        can_find_next = False
        for template in all_templates[:]:
            if template in model_important_template:
                continue
            all_templates.remove(template)
            x_tr = copy.deepcopy(original_x_tr)
            x_te = copy.deepcopy(original_x_te)
            for i in range(len(x_tr)):
                for item in x_tr[i][:]:
                    if item not in all_templates:
                        x_tr[i].remove(item)
            for i in range(len(x_te)):
                for item in x_te[i][:]:
                    if item not in all_templates:
                        x_te[i].remove(item)
            precision, recall, f1 = get_model_res(_model, x_tr, y_train, x_te, y_test)
            log_writer.write(str([precision, recall, f1]) + "\n")
            log_writer.flush()
            if f1 < (1 - f1_down_ratio) * max_f1:
                # important template and add it back
                all_templates.append(template)
                model_important_template.append(template)
            else:
                if max_f1 < f1:
                    max_precision = precision
                    max_recall = recall
                    max_f1 = f1
                    model_anti_template.append(template)
                can_find_next = True
                break
        if not can_find_next:
            break

    reduced_line = 0
    for i in range(len(x_data)):
        for item in x_data[i][:]:
            if item not in all_templates:
                x_data[i].remove(item)
        reduced_line += len(x_data[i])

    log_writer.write("template:" + str(all_templates) + "\n")
    log_writer.write("total_line:" + str(total_line) + "\n")
    log_writer.write("reduced_line:" + str(reduced_line) + "\n")
    log_writer.write("total_events:" + str(total_events) + "\n")
    log_writer.write("reduced_events:" + str(len(all_templates)) + "\n")
    log_writer.write("max_precision:" + str(max_precision) + "\n")
    log_writer.write("max_recall:" + str(max_recall) + "\n")
    log_writer.write("max_f1:" + str(max_f1) + "\n")
    log_writer.write("anti_template:" + str(model_anti_template) + "\n")
    log_writer.write("important_template:" + str(model_important_template) + "\n")
    log_writer.flush()
    log_writer.close()


def single_log(_model, CASE, node_index, events):
    train_ratio = 0.7
    (x_tr, y_train), (x_te, y_test) = dataloader.load_npz(
        f'../datasets/IoTDB/{CASE}/data{str(node_index)}.npz',
        train_ratio=train_ratio,
        split_type='uniform')
    for i in range(len(x_tr)):
        for item in x_tr[i][:]:
            if item not in events:
                x_tr[i].remove(item)
    for i in range(len(x_te)):
        for item in x_te[i][:]:
            if item not in events:
                x_te[i].remove(item)

    fw = open("test.log", "w")
    for i in range(len(x_te)):
        fw.write(str(x_te[i]) + "," + str(y_test[i]) + "\n")
    fw.flush()
    fw.close()

    precision, recall, f1 = get_model_res(_model, x_tr, y_train, x_te, y_test)
    print(_model, CASE, node_index, "FinalResult: ", [precision, recall, f1])


train_and_test_time("InvariantsMiner", "cpu_continue_leader", ["2"])

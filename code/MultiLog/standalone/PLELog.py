import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from utils.Vocab import Vocab
from datetime import datetime, timedelta

lstm_hiddens = 100
num_layer = 2
batch_size = 5
epochs = 50


class PLELog:
    _logger = logging.getLogger('PLELog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PLELog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for PLELog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return PLELog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_size = 128
        self.test_batch_size = 5
        self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def raw_predict(self, inputs):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits, dim=1)
        return tag_logits[:, 1].cpu().tolist()

    def batch_predict(self, vocab, instances):
        self.logger.info('Start batch predict by threshold None')
        with torch.no_grad():
            self.model.eval()
            result = []
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab, False)
                tinst.to_cuda(device)
                self.model.eval()
                try:
                    probs = self.raw_predict(tinst.inputs)
                    result.extend(probs)
                except Exception as e:
                    print("===batch_predict===error")
                    print(e)
                    pass
            return result

    def evaluate(self, vocab, processor, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab, False)
                tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, processor.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP + FP != 0 and TP + FN != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                end = time.time()
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f


def train_and_test_time(dataset, CASE, node_list, time_window_size):
    result_file = open(f'result_time_predict_PLELog_{CASE}.log', "w")

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
    min_cluster_size = 100
    min_samples = 100
    reduce_dimension = 50
    threshold = 0.5

    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')

    for node in node_list:
        result_file.write("===node:" + node + "===" + "\n")
        output_model_dir = os.path.join(save_dir,
                                        'models/PLELog/' + dataset + "_" + CASE + "_" + node + '_' + parser + '/model')
        prob_label_res_file = os.path.join(save_dir,
                                           'results/PLELog/' + dataset + '_' + CASE + "_" + node + '_' + parser +
                                           '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(
                                               min_samples))
        rand_state = os.path.join(save_dir,
                                  'results/PLELog/' + dataset + '_' + CASE + "_" + node + '_' + parser +
                                  '/prob_label_res/random_state')

        # Training, Validating and Testing instances.
        template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
        processor = Preprocessor()

        train, dev, instances, time_index2block = processor.process(dataset=dataset, case=CASE,
                                                                    node_index=node, parsing=parser,
                                                                    cut_func=cut_by_613,
                                                                    template_encoding=template_encoder.present,
                                                                    time_list=time_list)

        # Log sequence representation.
        sequential_encoder = Sequential_TF(processor.embedding)
        train_reprs = sequential_encoder.present(train)
        for index, inst in enumerate(train):
            inst.repr = train_reprs[index]
        # dev_reprs = sequential_encoder.present(dev)
        # for index, inst in enumerate(dev):
        #     inst.repr = dev_reprs[index]
        # test_reprs = sequential_encoder.present(test)
        # for index, inst in enumerate(test):
        #     inst.repr = test_reprs[index]

        # Dimension reduction if specified.
        if reduce_dimension != -1:
            start_time = time.time()
            print("Start FastICA, target dimension: %d" % reduce_dimension)
            transformer = FastICA(n_components=reduce_dimension)
            train_reprs = transformer.fit_transform(train_reprs)
            for idx, inst in enumerate(train):
                inst.repr = train_reprs[idx]
            print('Finished at %.2f' % (time.time() - start_time))

        # Probabilistic labeling.
        # Sample normal instances.
        train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
        normal_ids = train_normal[:int(0.5 * len(train_normal))]
        label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                                 res_file=prob_label_res_file, rand_state_file=rand_state)

        labeled_train = label_generator.auto_label(train, normal_ids)

        # Below is used to test if the loaded result match the original clustering result.
        TP, TN, FP, FN = 0, 0, 0, 0

        for inst in labeled_train:
            if inst.predicted == 'Normal':
                if inst.label == 'Normal':
                    TN += 1
                else:
                    FN += 1
            else:
                if inst.label == 'Anomalous':
                    TP += 1
                else:
                    FP += 1
        from utils.common import get_precision_recall

        print(len(normal_ids))
        print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
        p, r, f = get_precision_recall(TP, TN, FP, FN)
        print('%.4f, %.4f, %.4f' % (p, r, f))

        # Load Embeddings
        vocab = Vocab()
        vocab.load_from_dict(processor.embedding)

        plelog = PLELog(vocab, num_layer, lstm_hiddens, processor.label2id)

        log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
        last_model_file = os.path.join(output_model_dir, log + '_last.pt')
        if not os.path.exists(output_model_dir):
            os.makedirs(output_model_dir)

        # Train
        if not os.path.exists(last_model_file):
            optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
            global_step = 0
            batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

            for epoch in range(epochs):
                plelog.model.train()
                start = time.strftime("%H:%M:%S")
                plelog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                                   (epoch + 1, start, optimizer.lr))
                batch_iter = 0
                # start batch
                for onebatch in data_iter(labeled_train, batch_size, True):
                    try:
                        plelog.model.train()
                        tinst = generate_tinsts_binary_label(onebatch, vocab)
                        tinst.to_cuda(device)
                        loss = plelog.forward(tinst.inputs, tinst.targets)
                        loss_value = loss.data.cpu().numpy()
                        loss.backward()
                        if batch_iter % 100 == 0:
                            plelog.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                               % (global_step, epoch, batch_iter, loss_value))
                    except Exception as e:
                        print("train===error===")
                    batch_iter += 1
                    if batch_iter % 1 == 0 or batch_iter == batch_num:
                        nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, plelog.model.parameters()),
                            max_norm=1)
                        optimizer.step()
                        plelog.model.zero_grad()
                        global_step += 1
                plelog.logger.info('Training epoch %d finished.' % epoch)
                torch.save(plelog.model.state_dict(), last_model_file)

        TP, TN, FP, FN = 0, 0, 0, 0
        if os.path.exists(last_model_file):
            plelog.logger.info('=== Best Model ===')
            plelog.model.load_state_dict(torch.load(last_model_file))
            for time_index in range(len(time_list)):
                if time_index in time_index2block:
                    curr_instances = get_by_blocks(instances, time_index2block[time_index])
                    probs = plelog.batch_predict(vocab, curr_instances)
                    if len(probs) <= 0:
                        label = 0
                    elif max(probs) > 0.8:
                        label = 1
                    else:
                        label = 0
                    if label == 0:
                        if truth_label_list[time_index] == 0:
                            TN += 1
                        else:
                            FN += 1
                    else:
                        if truth_label_list[time_index] == 0:
                            FP += 1
                        else:
                            TP += 1
                    result_file.write(str(time_index) + ":" + str(probs) + "\n")
                    result_file.flush()
                else:
                    result_file.write(str(time_index) + ":" + str([]) + "\n")
                    result_file.flush()
        precision = 100 * TP / (TP + FP)
        recall = 100 * TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(precision, recall, f1))
    print('All Finished')
    result_file.close()
    return [0, 0, 0]


def train_and_test_time_single_node(dataset, CASE, NODE, time_window_size):
    result_file = open(f'result_time_predict_PLELog_{CASE}_{NODE}.log', "w")

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
    min_cluster_size = 100
    min_samples = 100
    reduce_dimension = 50
    threshold = 0.5

    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')

    result_file.write("===node:" + NODE + "===" + "\n")
    output_model_dir = os.path.join(save_dir,
                                    'models/PLELog/' + dataset + "_" + CASE + "_" + NODE + '_' + parser + '/model')
    prob_label_res_file = os.path.join(save_dir,
                                       'results/PLELog/' + dataset + '_' + CASE + "_" + NODE + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(
                                           min_samples))
    rand_state = os.path.join(save_dir,
                              'results/PLELog/' + dataset + '_' + CASE + "_" + NODE + '_' + parser +
                              '/prob_label_res/random_state')

    # Training, Validating and Testing instances.
    template_encoder = Template_TF_IDF_without_clean() if dataset == 'NC' else Simple_template_TF_IDF()
    processor = Preprocessor()

    train, dev, instances, time_index2block = processor.process(dataset=dataset, case=CASE,
                                                                node_index=NODE, parsing=parser,
                                                                cut_func=cut_by_613,
                                                                template_encoding=template_encoder.present,
                                                                time_list=time_list)

    # Log sequence representation.
    sequential_encoder = Sequential_TF(processor.embedding)
    train_reprs = sequential_encoder.present(train)
    for index, inst in enumerate(train):
        inst.repr = train_reprs[index]

    # Dimension reduction if specified.
    if reduce_dimension != -1:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer = FastICA(n_components=reduce_dimension)
        train_reprs = transformer.fit_transform(train_reprs)
        for idx, inst in enumerate(train):
            inst.repr = train_reprs[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    # Probabilistic labeling.
    # Sample normal instances.
    train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:int(0.5 * len(train_normal))]
    label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                             res_file=prob_label_res_file, rand_state_file=rand_state)

    labeled_train = label_generator.auto_label(train, normal_ids)

    # Below is used to test if the loaded result match the original clustering result.
    TP, TN, FP, FN = 0, 0, 0, 0

    for inst in labeled_train:
        if inst.predicted == 'Normal':
            if inst.label == 'Normal':
                TN += 1
            else:
                FN += 1
        else:
            if inst.label == 'Anomalous':
                TP += 1
            else:
                FP += 1
    from utils.common import get_precision_recall

    print(len(normal_ids))
    print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
    p, r, f = get_precision_recall(TP, TN, FP, FN)
    print('%.4f, %.4f, %.4f' % (p, r, f))

    # Load Embeddings
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    plelog = PLELog(vocab, num_layer, lstm_hiddens, processor.label2id)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    # Train
    if not os.path.exists(last_model_file):
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        global_step = 0
        batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

        for epoch in range(epochs):
            plelog.model.train()
            start = time.strftime("%H:%M:%S")
            plelog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            # start batch
            for onebatch in data_iter(labeled_train, batch_size, True):
                plelog.model.train()
                tinst = generate_tinsts_binary_label(onebatch, vocab)
                tinst.to_cuda(device)
                loss = plelog.forward(tinst.inputs, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward()
                if batch_iter % 100 == 0:
                    plelog.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                       % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, plelog.model.parameters()),
                        max_norm=1)
                    optimizer.step()
                    plelog.model.zero_grad()
                    global_step += 1
            plelog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(plelog.model.state_dict(), last_model_file)

    if os.path.exists(last_model_file):
        plelog.logger.info('=== Best Model ===')
        plelog.model.load_state_dict(torch.load(last_model_file))
        for time_index in range(len(time_list)):
            if time_index in time_index2block:
                curr_instances = get_by_blocks(instances, time_index2block[time_index])
                label = plelog.batch_predict(vocab, curr_instances)
                result_file.write(str(time_index) + ":" + str(label) + "\n")
                result_file.flush()
            else:
                result_file.write(str(time_index) + ":" + str([]) + "\n")
                result_file.flush()

    result_file.close()
    return [0, 0, 0]


def get_by_blocks(instances, block_list):
    result = []
    for instance in instances:
        if instance.id in block_list:
            result.append(instance)
    return result

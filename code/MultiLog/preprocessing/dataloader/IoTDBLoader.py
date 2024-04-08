import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader
from datetime import datetime, timedelta


class IoTDBLoader(BasicDataLoader):
    def __init__(self, in_file=None,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/IoTDB'),
                 semantic_repr_func=None,
                 time_list=None):
        super(IoTDBLoader, self).__init__()

        # Construct logger.
        self.logger = logging.getLogger('IoTDBLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'IoTDBLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct self.logger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.remove_cols = []
        self.dataset_base = dataset_base
        self.time_list = time_list
        self._load_raw_log_seqs()
        self.semantic_repr_func = semantic_repr_func
        pass

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for id, token in enumerate(tokens):
            if id not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)
        # return re.sub('[\*\.\?\+\$\^\[\]\(\)\{\}\|\\\/]', '', ' '.join(after_process))

    def _load_raw_log_seqs(self):
        self.logger.info('_load_raw_log_seqs')
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        time_block_file = os.path.join(self.dataset_base, 'time_block.txt')
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(':')
                    self.block2label[block_id] = label
            with open(time_block_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    time_index, block_idx_list_str = line.strip().split(':')
                    self.time_index2block[int(time_index)] = block_idx_list_str.split(",")
        else:
            self.logger.info('Start loading IoTDB log sequences.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                self.logger.info("lineNum:" + str(len(lines)))
                log_start_time = 999999999999999
                log_end_time = 0
                nodes = OrderedDict()
                for idx, line in enumerate(lines):
                    tokens = line.strip().split()
                    if line.startswith('-'):
                        node = str(tokens[3])
                    else:
                        node = str(tokens[2])
                    if "$" in node:
                        node = node.split("$")[0]
                    if node not in nodes.keys():
                        nodes[node] = []
                    time_str = line.split("[")[0][:-1].replace("- ", "")
                    idx_time = (datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f") + timedelta(hours=8)).timestamp()
                    nodes[node].append((idx, line.strip(), idx_time))
                    log_start_time = min(idx_time, log_start_time)
                    log_end_time = max(idx_time, log_end_time)
                self.logger.info("log_start_time:" + str(log_start_time))
                self.logger.info("log_end_time:" + str(log_end_time))
                self.logger.info("idx:" + str(idx))

                nodes[node] = sorted(nodes[node], key=lambda x: x[2])

                pbar = tqdm(total=len(nodes))

                block_idx = 0
                for node, seq in nodes.items():
                    label = 'Normal'
                    block_list = []
                    curr_time_index = 0
                    if curr_time_index not in self.time_index2block:
                        self.time_index2block[curr_time_index] = []
                    for (idx, line, idx_time) in seq:
                        if self.time_list[curr_time_index][0] <= idx_time <= self.time_list[curr_time_index][
                            1]:
                            block_list.append(idx)
                            if not line.startswith('-'):
                                label = 'Anomalous'
                        elif idx_time > self.time_list[curr_time_index][1]:
                            if len(block_list) > 0:
                                self.blocks.append(str(block_idx))
                                self.block2seqs[str(block_idx)] = block_list
                                self.block2label[str(block_idx)] = label
                                self.time_index2block[curr_time_index].append(str(block_idx))
                                block_idx += 1

                            while idx_time > self.time_list[curr_time_index][1]:
                                curr_time_index += 1
                                if curr_time_index not in self.time_index2block:
                                    self.time_index2block[curr_time_index] = []
                            label = 'Normal'
                            block_list = []
                        else:
                            self.logger.info("error condition")
                            self.logger.info(idx_time)
                            self.logger.info(line)
                    if len(block_list) > 0:
                        self.blocks.append(str(block_idx))
                        self.block2seqs[str(block_idx)] = block_list
                        self.block2label[str(block_idx)] = label
                        self.time_index2block[curr_time_index].append(str(block_idx))
                        block_idx += 1
                    pbar.update(1)
                pbar.close()

            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

            with open(time_block_file, 'w', encoding='utf-8') as writer:
                for time_index in self.time_index2block.keys():
                    writer.write(str(time_index) + ':' + ','.join(self.time_index2block[time_index]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass


if __name__ == '__main__':
    from representations.templates.statistics import Simple_template_TF_IDF

    semantic_encoder = Simple_template_TF_IDF()
    loader = IoTDBLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/IoTDB/IoTDB.log'),
                         dataset_base=os.path.join(PROJECT_ROOT, 'datasets/IoTDB'),
                         semantic_repr_func=semantic_encoder.present)
    loader.parse_by_IBM(config_file=os.path.join(PROJECT_ROOT, 'conf/IoTDB.ini'),
                        persistence_folder=os.path.join(PROJECT_ROOT, 'datasets/IoTDB/persistences'))
    pass

import sys

sys.path.extend([".",".."])
from CONSTANTS import *
from collections import OrderedDict
from preprocessing.BasicLoader import BasicDataLoader


class ThunderbirdLoader(BasicDataLoader):
    def __init__(self, in_file=None,
                 window_size=120,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/Thunderbird'),
                 semantic_repr_func=None):
        super(ThunderbirdLoader, self).__init__()

        # Construct logger.
        self.logger = logging.getLogger('ThunderbirdLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'ThunderbirdLoader.log'))
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
        self.window_size = window_size
        self.dataset_base = dataset_base
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

    def parse_by_Official(self):
        self._restore()
        # Define official templates
        templates = []

        save_path = os.path.join(PROJECT_ROOT, 'datasets/Thunderbird/persistences/official')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        templates_file = os.path.join(save_path, 'NC_templates.txt')
        log2temp_file = os.path.join(save_path, 'log2temp.txt')
        if os.path.exists(templates_file) and os.path.exists(log2temp_file):
            self.logger.info('Found parsing result, please note that this does not guarantee a smooth execution.')
            with open(templates_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(',')
                    id = int(tokens[0])
                    template = ','.join(tokens[1:])
                    self.templates[id] = template

            with open(log2temp_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    logid, tempid = line.strip().split(',')
                    self.log2temp[int(logid)] = int(tempid)

            pass

        else:
            for id, template in enumerate(templates):
                self.templates[id] = template
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    if self.remove_cols:
                        processed_line = self._pre_process(line)
                    for index, template in self.templates.items():
                        if re.compile(template).match(processed_line) is not None:
                            self.log2temp[log_id] = index
                            break
                    if log_id not in self.log2temp.keys():
                        # if processed_line == '':
                        #     self.log2temp[log_id] = -1
                        self.logger.warning('Mismatched log message: %s' % processed_line)
                        for index, template in self.templates.items():
                            if re.compile(template).match(line) is not None:
                                self.log2temp[log_id] = index
                                break
                        if log_id not in self.log2temp.keys():
                            self.logger.error('Failed to parse line %s' % line)
                            exit(2)
                    log_id += 1

            with open(templates_file, 'w', encoding='utf-8') as writer:
                for id, template in self.templates.items():
                    writer.write(','.join([str(id), template]) + '\n')
            with open(log2temp_file, 'w', encoding='utf-8') as writer:
                for logid, tempid in self.log2temp.items():
                    writer.write(','.join([str(logid), str(tempid)]) + '\n')
            # with open(logseq_file, 'w', encoding='utf-8') as writer:
            #     self._save_log_event_seqs(writer)
        self._prepare_semantic_embed(os.path.join(save_path, 'event2semantic.vec'))
        # Summarize log event sequences.
        for block, seq in self.block2seqs.items():
            self.block2eventseq[block] = []
            for log_id in seq:
                self.block2eventseq[block].append(self.log2temp[log_id])

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
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

        else:
            self.logger.info('Start loading Thunderbird log sequences.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                lines = reader.readlines()
                nodes = OrderedDict()
                for idx, line in enumerate(lines):
                    tokens = line.strip().split()
                    node = str(tokens[3])
                    if node not in nodes.keys():
                        nodes[node] = []
                    nodes[node].append((idx, line.strip()))

                pbar = tqdm(total=len(nodes))

                block_idx = 0
                for node, seq in nodes.items():
                    if len(seq) < self.window_size:
                        self.blocks.append(str(block_idx))
                        self.block2seqs[str(block_idx)] = []
                        label = 'Normal'
                        for (idx, line) in seq:
                            self.block2seqs[str(block_idx)].append(int(idx))
                            if not line.startswith('-'):
                                label = 'Anomalous'
                        self.block2label[str(block_idx)] = label
                        block_idx += 1
                    else:
                        i = 0
                        while i < len(seq):
                            self.blocks.append(str(block_idx))
                            self.block2seqs[str(block_idx)] = []
                            label = 'Normal'
                            for (idx, line) in seq[i:i + self.window_size]:
                                self.block2seqs[str(block_idx)].append(int(idx))
                                if not line.startswith('-'):
                                    label = 'Anomalous'
                            self.block2label[str(block_idx)] = label
                            block_idx += 1
                            i += self.window_size

                    pbar.update(1)

                pbar.close()
            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass


if __name__ == '__main__':
    from representations.templates.statistics import Simple_template_TF_IDF

    semantic_encoder = Simple_template_TF_IDF()
    loader = ThunderbirdLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/Thunderbird/Thunderbird.log'),
                       dataset_base=os.path.join(PROJECT_ROOT, 'datasets/Thunderbird'),
                       semantic_repr_func=semantic_encoder.present)
    loader.parse_by_IBM(config_file=os.path.join(PROJECT_ROOT, 'conf/Thunderbird.ini'),
                        persistence_folder=os.path.join(PROJECT_ROOT, 'datasets/Thunderbird/persistences'))
    pass

import pickle
import random
import json, os
from transformers import BertTokenizer
import numpy as np
def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    return tokenizer


class data_sampler(object):

    def __init__(self, args, seed=None):
        self.set_path(args)
        self.args = args
        temp_name = [args.dataname, args.seed]
        file_name = "{}.pkl".format(
                    "-".join([str(x) for x in temp_name])
                )
        mid_dir = "datasets/"
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        for temp_p in ["_process_path"]:
            mid_dir = os.path.join(mid_dir, temp_p)
            if not os.path.exists(mid_dir):
                os.mkdir(mid_dir)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(args)

        # read relation data
        self.id2rel, self.rel2id = self._read_relations(args.relation_file)

        # random sampling
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

        # regenerate data
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.args.data_file)

        # generate the task number
        self.batch = 0
        self.task_length = len(self.id2rel) // self.args.rel_per_task

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
    def set_path(self, args):
        use_marker = ""
        if args.dataname in ['FewRel']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel.json")
            args.num_of_relation = 80
            args.num_of_train = 420
            args.num_of_val = 140
            args.num_of_test = 140
        elif args.dataname in ['TACRED']:
            args.data_file = os.path.join(args.data_path,"data_with{}_marker_tacred.json".format(use_marker))
            args.relation_file = os.path.join(args.data_path, "id2rel_tacred.json")
            args.num_of_relation = 40
            args.num_of_train = 420
            args.num_of_val = 140
            args.num_of_test = 140

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)

    def __iter__(self):
        return self

    def __next__(self):

        if self.batch == self.task_length:
            raise StopIteration()

        indexs = self.shuffle_index[self.args.rel_per_task*self.batch: self.args.rel_per_task*(self.batch+1)]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}
        
        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])
            cur_training_data[self.id2rel[index]] = self.training_dataset[index]
            cur_valid_data[self.id2rel[index]] = self.valid_dataset[index]
            cur_test_data[self.id2rel[index]] = self.test_dataset[index]
            self.history_test_data[self.id2rel[index]] = self.test_dataset[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations

    def _read_data(self, file):
        if os.path.isfile(self.save_data_path):
            with open(self.save_data_path, 'rb') as f:
                datas = pickle.load(f)
            train_dataset, val_dataset, test_dataset = datas
            return train_dataset, val_dataset, test_dataset
        else:
            data = json.load(open(file, 'r', encoding='utf-8'))
            train_dataset = [[] for i in range(self.args.num_of_relation)]
            val_dataset = [[] for i in range(self.args.num_of_relation)]
            test_dataset = [[] for i in range(self.args.num_of_relation)]
            for relation in data.keys():
                rel_samples = data[relation]
                if self.seed != None:
                    random.seed(self.seed)
                random.shuffle(rel_samples)
                count = 0
                count1 = 0
                for i, sample in enumerate(rel_samples):
                    tokenized_sample = {}
                    tokenized_sample['relation'] = self.rel2id[sample['relation']]
                    tokenized_sample['tokens'] = self.tokenizer.encode(' '.join(sample['tokens']),
                                                                    padding='max_length',
                                                                    truncation=True,
                                                                    max_length=self.args.max_length)
                    if self.args.task_name == 'FewRel':
                        if i < self.args.num_of_train:
                            train_dataset[self.rel2id[relation]].append(tokenized_sample)
                        elif i < self.args.num_of_train + self.args.num_of_val:
                            val_dataset[self.rel2id[relation]].append(tokenized_sample)
                        else:
                            test_dataset[self.rel2id[relation]].append(tokenized_sample)
                    else:
                        if i < len(rel_samples) // 5 and count <= 40:
                            count += 1
                            test_dataset[self.rel2id[relation]].append(tokenized_sample)
                        else:
                            count1 += 1
                            train_dataset[self.rel2id[relation]].append(tokenized_sample)  
                            if count1 >= 320:
                                break
            with open(self.save_data_path, 'wb') as f:
                pickle.dump((train_dataset, val_dataset, test_dataset), f)
            return train_dataset, val_dataset, test_dataset

    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        rel2id = {}
        for i, x in enumerate(id2rel):
            rel2id[x] = i
        return id2rel, rel2id
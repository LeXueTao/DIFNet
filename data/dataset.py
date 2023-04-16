import os
import numpy as np
import itertools
import collections
import torch
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO
import json
import time
import random


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)
        
    def collate_fn(self):
        def collate(batch):
            # cpu_time_start = time.time()
            if len(self.fields) == 1:
                batch = [batch, ]
            else: # 图片和图片放一起，文字和文字放一起
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data) #这一步转换为tensor
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)
            # cpu_time_end = time.time()
            # print("collate_time:{}".format(cpu_time_end- cpu_time_start))
            # 返回List，里面是3个类型，(batch_size, dim)
            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        start_time = time.time()
        for field_name, field in self.fields.items():
            start_time_1 = time.time()
            data.append(field.preprocess(getattr(example, field_name)))
            end_time_1 = time.time()
            # print(field_name+"_time:{}".format(start_time_1-end_time_1))
        end_time = time.time()
        # print("data_item:{}".format(end_time-start_time))
        if len(data) == 1: # 兼容性
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)
    # 返回一个迭代器，节省空间
    def __getattr__(self, attr): 
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    """处理gt_cap数据，返回一个照片对应的所有cap"""
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    """dict数据集用于SCST训练和各种分数测试"""
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)
        # 不存在key返回空列表
        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        assert ('pixel' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']
        self.pixel_field = self.fields['pixel']

    def image_set(self):
        # 加载了两个val集合，这里是去重复
        # 很遗憾，没在文件中找到该函数的使用地方
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        # 单独生成Image_dataset
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        # 加载了两个val集合，这里是去重复
        # 由于pixel的在提取时候去过了，这里不用去重
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields=['image', 'pixel'])
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    """COCO只负责了提取划分"""
    def __init__(self, image_field, text_field, pixel_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }


        ids = None
        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {'image': image_field, 'text': text_field, 'pixel': pixel_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        all_samples = []
        train_samples = []
        val_samples = []
        test_samples = []

        # 虽然开头给test的地址是val的地址
        # 按照karpathy划分
        # val:5000
        # test: 5000
        # train: 余下全部
        
        # 加载coco_ann
        coco_train = pyCOCO(roots['train']['cap'])
        root_train = roots['train']['img']
        ann_ids_train = list(coco_train.anns.keys())
        coco_val = pyCOCO(roots['val']['cap'])
        root_val = roots['val']['img']
        ann_ids_val = list(coco_val.anns.keys())

        # 取训练集数据
        for index in ann_ids_train:
            caption = coco_train.anns[index]['caption']
            img_id = coco_train.anns[index]['image_id']
            filename = coco_train.loadImgs(img_id)[0]['file_name']
            # 生成一个类，类属性是这三个东西
            example = Example.fromdict({'image': os.path.join( root_train, filename), 'text': caption, 'pixel': os.path.join( root_train, filename)})
            all_samples.append(example)
        
        # 取测试集集数据
        for index in ann_ids_val:
            caption = coco_val.anns[index]['caption']
            img_id = coco_val.anns[index]['image_id']
            filename = coco_val.loadImgs(img_id)[0]['file_name']
            # 生成一个类，类属性是这三个东西
            example = Example.fromdict({'image': os.path.join(root_val, filename), 'text': caption, 'pixel': os.path.join(root_val, filename)})
            all_samples.append(example)

        # 打乱数据
        random.shuffle(all_samples)
        val_samples = all_samples[0:5000]
        test_samples = all_samples[5000:10000]
        train_samples = all_samples[10000:]        

        return train_samples, val_samples, test_samples


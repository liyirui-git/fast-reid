# encoding: utf-8
"""
@author:  iry-lee
@contact: liyiruiasp@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BikePerson (ImageDataset):
    '''Bike-Person

    Reference:
        Yuan et al. Bike-Person Re-identification: A Benchmark and A Comprehensive Evaluation. IEEE Access 2016.
    
    URL: `<https://drive.google.com/file/d/1u6906LTa2xU4fibwfqkT6cEn81fjOFJa/view>`_

    Dataset statistics:
        - identities: 4579
    '''

    # _junk_pids 是啥不清楚 
    _junk_pids = [0, -1]
    dataset_dir = 'BikePersonDataset-700-seg'
    dataset_url = ''
    dataset_name = 'bikeperson'
    
    # market1501.__init__ 还有两个参数 market1501_500k = False, **kwargs 不清楚啥意思
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        
        super(BikePerson, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg')) + glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
        

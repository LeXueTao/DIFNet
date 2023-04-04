'''
特征文件过大，无法直接放入百度网盘，这里将每个特征文件拆成若干个子文件split_files，每个子文件包含10k个图像的特征。
如果您要使用这个特征，请执行merge_files来合并子文件。
'''

import os
import h5py
from tqdm import tqdm


feat_root = './coco2014_gridfeats'             # 完整文件位置
target_root = './coco2014_gridfeats'           # 子文件位置，百度网盘上下载的文件置于此处
file_list = ['X101_grid_feats_coco_trainval.hdf5', 'X101_grid_feats_coco_test.hdf5', 'X152_grid_feats_coco_trainval.hdf5', 'X152_grid_feats_coco_test.hdf5']
assert os.path.exists(feat_root)
os.makedirs(target_root, exist_ok=True)


# 拆分特征文件
def split_files(backbone='X152', split='trainval'):

    def save_file(f, name, per_num=10000):
        keys = list(f.keys())
        file_num = len(keys)
        print('file_num: ', file_num)
        idx = 0
        start = 0

        feat_dir = os.path.join(target_root, name)
        os.makedirs(feat_dir, exist_ok=True)

        while start + per_num < file_num:
            print('processing {}'.format(idx))
            end = start + per_num
            cur_keys = keys[start:end]
            file_path = os.path.join(feat_dir, name+'_'+str(idx)+'.hdf5')
            with h5py.File(file_path, 'w') as f_w:
                for k in tqdm(range(len(cur_keys))):
                    key = cur_keys[k]
                    img_feat = f[key][()]
                    # 保存特征
                    f_w.create_dataset(key, data=img_feat)
            f_w.close()

            start = end
            idx += 1
        
        print('processing {}'.format(idx))
        file_path = os.path.join(feat_dir, name+'_'+str(idx)+'.hdf5')
        print('start: ', start)
        cur_keys = keys[start:file_num]
        with h5py.File(file_path, 'w') as f_w:
            for k in tqdm(range(len(cur_keys))):
                key = cur_keys[k]
                img_feat = f[key][()]
                # 保存特征
                f_w.create_dataset(key, data=img_feat) 
        f_w.close()
        print('{} is done!'.format(name))
    
    if backbone == 'X101' and split == 'trainval':
        filename = 'X101_grid_feats_coco_trainval.hdf5'
    elif backbone == 'X101' and split == 'test':
        filename = 'X101_grid_feats_coco_test.hdf5'
    elif backbone == 'X152' and split == 'trainval':
        filename = 'X152_grid_feats_coco_trainval.hdf5'
    elif backbone == 'X152' and split == 'test':
        filename = 'X152_grid_feats_coco_test.hdf5'
    else:
        raise ValueError('backbone {} - split {} is nos provided'.format(backbone, split))

    filepath = os.path.join(feat_root, filename)
    print('this is {}'.format(filename))
    f = h5py.File(filepath, 'r')
    
    name = backbone + '_' + split       # name = 'X152_trainval'

    save_file(f, name)

    # print(keys[:5])
    # ['100004_grids', '100007_grids', '100030_grids', '100035_grids', '100051_grids']


# 合并特征文件
def merge_files(backbone='X152', split='trainval'):
    # 确定子文件位置
    name = backbone + '_' + split       # name = 'X152_trainval'
    feat_dir = os.path.join(target_root, name)
    featnames = os.listdir(feat_dir)
    featpaths = [os.path.join(feat_dir, featname) for featname in featnames]

    if backbone == 'X101' and split == 'trainval':
        filename = 'X101_grid_feats_coco_trainval.hdf5'
    elif backbone == 'X101' and split == 'test':
        filename = 'X101_grid_feats_coco_test.hdf5'
    elif backbone == 'X152' and split == 'trainval':
        filename = 'X152_grid_feats_coco_trainval.hdf5'
    elif backbone == 'X152' and split == 'test':
        filename = 'X152_grid_feats_coco_test.hdf5'
    else:
        raise ValueError('backbone {} - split {} is nos provided'.format(backbone, split))

    file_path = os.path.join(filename)
    with h5py.File(file_path, 'w') as f_w:      # 写入完整文件
        for featpath in featpaths:
            # 读取子文件
            f = h5py.File(featpath, 'r') 
            keys = list(f.keys())
            for k in tqdm(range(len(keys))):
                key = keys[k]
                img_feat = f[key][()]
                # 保存特征
                f_w.create_dataset(key, data=img_feat)
    f_w.close()


# 核对文件数目
def check(backbone='X152', split='trainval'):
    name = backbone + '_' + split       # name = 'X152_trainval'
    feat_dir = os.path.join(target_root, name)
    featnames = os.listdir(feat_dir)
    featpaths = [os.path.join(feat_dir, featname) for featname in featnames]
    print(featpaths)
    total_keys = []
    for file in featpaths:
        f = h5py.File(file, 'r')
        keys = list(f.keys())
        key = keys[10]
        data = f[key][()]
        print(len(keys), key, data.shape)
        total_keys += keys
    total_keys = list(set(total_keys))
    print(len(total_keys))
    

if __name__ == '__main__':
    '''
    trainval: 123287
    test: 40775
    '''
    backbone = 'X101'
    split = 'trainval'
    # split_files(backbone=backbone, split=split)
    merge_files(backbone=backbone, split=split)
    check(backbone=backbone, split=split)
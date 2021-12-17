import logging
import os
import glob

import torch.utils.data
from util.pointcloud import make_open3d_point_cloud, get_matching_indices
from dataloader.base_loader import *
from dataloader.transforms import *
from dataloader.iralab_base_loader import IRALABBenchmarkPairDataset
import numpy as np
import open3d as o3d
import pandas as pd


class IralabBenchmarkTestLoader(IRALABBenchmarkPairDataset):

    def __init__(self, run, path, is_local=True, config=None, transform=None):

        if is_local:
            self.test_file = os.path.join(path, run) + "_local.txt"
        else:
            self.test_file = os.path.join(path, run) + "_global.txt"

        self.point_cloud_path = cloud_path = os.path.join(path, run)
        self.path_to_dataset = path
        self.is_local = is_local
        self.config = config
        self.voxel_size = config.voxel_size
        IRALABBenchmarkPairDataset.__init__(self, run=run, test_file=self.test_file, config=config, transform=transform)
        logging.info(f"Loading the experiment {run} from {cloud_path}")

    def __getitem__(self, id):
        test_id, source_fn, target_fn = self.get_pair_fn(id)

        source_pcd = o3d.io.read_point_cloud(source_fn)
        target_pcd = o3d.io.read_point_cloud(target_fn)
        xyz0 = np.asarray(source_pcd.points)
        xyz1 = np.asarray(target_pcd.points)

        tf, itf = self.get_tf(id)
        xyz0 = self.apply_transform(xyz0, tf)

        # work on the downsampled xyzs, 0.05m == 5cm
        #_, seldw0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
        #_, seldw1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        pcd0 = make_open3d_point_cloud(xyz0[sel0])
        pcd1 = make_open3d_point_cloud(xyz1[sel1])

        matching_search_voxel_size = self.matching_search_voxel_size
        # Get matches

        matches = get_matching_indices(pcd0, pcd1, itf, matching_search_voxel_size)
        if len(matches) < 1000:
            raise ValueError(f"Insufficient matches in {self.run}, {tf}, {itf}")

        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        if self.transform:
            coords0, feats0 = self.transform(coords0, feats0)
            coords1, feats1 = self.transform(coords1, feats1)

        extra_package = {'run': self.run, 'tf': tf, 'itf': itf}

        return (unique_xyz0_th.float(),
                unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
                feats1.float(), matches, tf, itf, extra_package)


    def get_pair_fn(self, id):
        test_id = self.files[id][0]
        source_fn = self.files[id][1]
        target_fn = self.files[id][2]

        source_fn = self.path_to_dataset + "/" + self.run + "/" + source_fn
        target_fn = self.path_to_dataset + "/" + self.run + "/" + target_fn

        return test_id, source_fn, target_fn


    def get_tf(self, id):
        tf = np.eye(4)
        itf = np.eye(4)

        R = np.asarray([[self.dataframe.iloc[id]['t1'], self.dataframe.iloc[id]['t2'], self.dataframe.iloc[id]['t3']],
             [self.dataframe.iloc[id]['t5'], self.dataframe.iloc[id]['t6'], self.dataframe.iloc[id]['t7']],
             [self.dataframe.iloc[id]['t9'], self.dataframe.iloc[id]['t10'], self.dataframe.iloc[id]['t11']]])
        T = np.asarray([self.dataframe.iloc[id]['t4'], self.dataframe.iloc[id]['t8'], self.dataframe.iloc[id]['t12']])

        iR = np.linalg.inv(R)
        iT = -T

        tf[:3, :3] = R
        tf[:3, 3] = T.T

        itf[:3, :3] = iR
        itf[:3, 3] = iT.T

        return tf, itf

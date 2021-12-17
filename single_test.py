import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
import numpy as np
from config import get_config
from dataloader.transforms import rand_trans
from core.metrics import rotation_error, translation_error


def invert(gt):
    igt = np.eye(4)
    igt[:3, :3] = np.linalg.inv(gt[:3, :3])
    igt[:3, 3] = -gt[:3, 3]
    return igt


class SingleTest:
    """
    config file
    deep global registration
    point cloud 0 (dst, target)
    point cloud 1 (source)
    gt [4x4]
    igt [4x4]
    """
    def __init__(self, config=None, dgr=None, pcd0=None, pcd1=None, gt=None):

        if dgr is not None:
            self.dgr = dgr
            if config is not None:
                self.config = config
        else:
            if config is not None:
                self.config = config
                self.dgr = DeepGlobalRegistration(self.config)
            else:
                assert False, "Init Error, Cannot initialize DeepGlobalRegistration"

        if gt is not None:
            self.gt = gt
            self.igt = invert(gt)
        else:
            assert False, "Init Error, set gt"

        if pcd0 is not None:
            self.pcd0 = pcd0
            if pcd1 is not None:
                self.pcd1 = pcd1
            else:
                self.pcd1 = self.pcd0.transform(self.igt)
        else:
            assert False, "Init Error, no point cloud found"

    def test(self):

        # preprocessing
        self.pcd0.estimate_normals()
        self.pcd1.estimate_normals()


        # registration
        result = self.dgr.register(self.pcd1, self.pcd0)
        self.pcd1 = self.pcd0.transform(result)

        r_err = rotation_error(result[:3, :3], self.gt[:3, :3])
        t_err = translation_error(result[:3, 3], self.gt[:3, 3])

        print("Rotation error: ", r_err)
        print("Translation error: ", t_err)


if __name__ == "__main__":
    config = get_config()

    if config.weights is None:
        config.weights = "ResUNetBN2C-feat32-3dmatch-v0.05.pth"

    cloud0 = o3d.io.read_point_cloud(config.pcd0)
    cloud1 = o3d.io.read_point_cloud(config.pcd1)

    print(np.asarray(cloud0.points).shape)

    gt = rand_trans(cloud0, np.random, rotation_range=180)

    test = SingleTest(config=config, pcd0=cloud0, pcd1=cloud1, gt=gt)
    test.test()

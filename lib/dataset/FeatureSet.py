import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    '''In ExemplarDataset, generally don't need target_transform. if exemplar is original images, need img_transform and 
    inv_transform else don't need'''

    def __init__(self, feature_sets, target_transform=None):
        super().__init__()
        self.feature_sets = feature_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.feature_sets)):
            total += len(self.feature_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        class_id = 0
        feature_id = 0
        class_id_to_return = None
        for class_id in range(len(self.feature_sets)):
            features_in_this_class = len(self.feature_sets[class_id])
            if index < (total + features_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                feature_id = index - total
                break
            else:
                total += features_in_this_class
        feature = torch.from_numpy(self.feature_sets[class_id][feature_id])
        return feature, class_id_to_return
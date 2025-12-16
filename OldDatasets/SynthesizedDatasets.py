from torch.utils.data import Dataset


class SynthesizedDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    '''In ExemplarDataset, generally don't need target_transform. if exemplar is original images, need img_transform and 
    inv_transform else don't need'''

    def __init__(self, synthesized_sets):
        super().__init__()
        self.synthesized_sets = synthesized_sets

    def __len__(self):
        total = 0
        for class_id in range(len(self.synthesized_sets)):
            total += len(self.synthesized_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.synthesized_sets)):
            exemplars_in_this_class = len(self.synthesized_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = self.synthesized_sets[class_id][exemplar_id]
        return image, class_id_to_return

from torch.utils.data import Dataset
import numpy as np


class DDC_dataset(Dataset):
    def __init__(self, dataset, mode="train", cfg=None, active_classes_num=None):
        self.dataset = dataset
        self.mode = mode
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))

        self.num_classes = active_classes_num

        self.num_class_list = np.array(self.get_cls_num_list())

        class_cate_index = np.zeros(self.num_classes).astype(np.int)  # few
        class_cate_index[self.num_class_list >= 20] = 1  # medium
        class_cate_index[self.num_class_list > 100] = 2  # many
        self.class_cate_index = class_cate_index

        if self.cfg.CLASSIFIER.TYPE == 'LDA' and self.mode == 'train':
            self.class_weight_for_lda = self.get_class_weight_for_lda(self.num_class_list)

    def get_class_weight_for_lda(self, img_num_list):
        cls_num_list = np.array(img_num_list)
        return 1 / (self.num_classes * cls_num_list)

    def __getitem__(self, index):

        img, img_label = self.dataset[index]
        meta = dict()
        # image_label = (
        #     now_info[1] if "test" not in self.mode else 0
        # )  # 0-index

        if self.cfg.CLASSIFIER.TYPE == 'LDA' and self.mode == 'train':
            meta['class_weight'] = self.class_weight_for_lda[img_label]
        meta['shot_cate'] = self.class_cate_index[img_label]

        return img, img_label, meta

    def __len__(self):
        return len(self.dataset)

    def get_num_classes(self):
        return self.num_classes

    def get_cls_num_list(self):
        cls_num_list = [0, ] * self.num_classes
        for d in self.dataset:
            cls_num_list[d[1]] += 1
        return cls_num_list

import copyreg
import types
import numpy as np

__all__ = ['Metrics']

class Metrics(object):
    def __init__(self, num_class):
        self.__num_class = num_class
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)

        self.__TP = 0.#np.diag(self.__confusion_matrix)
        self.__RealN = 0.#np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = 0.#np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = 0.#np.sum(self.__confusion_matrix)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.__confusion_matrix).sum() / self.__confusion_matrix.sum()
        return Acc

    def Class_Precision(self):
        #TP/TP+FP
        precision = self.__TP / (self.__RealP + 1e-5)
        # Acc = np.nanmean(Acc)
        return precision

    def Intersection_over_Union(self):
        IoU = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        return IoU

    def Mean_Intersection_over_Union(self):
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.__RealP / self.__sum
        iu = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        P0 = np.sum(self.__TP) / self.__sum
        Pe = np.sum(self.__RealP * self.__RealN) / (self.__sum * self.__sum)
        return (P0 - Pe) / (1 - Pe)

    def F1_score(self, belta=1):
        precision = self.Class_Precision()
        recall = self.Recall()
        f1_score = (1 + belta * belta) * precision * recall / (belta * belta * precision + recall + 1e-5)
        return f1_score

    def Macro_F1(self, belta=1):
        return np.nanmean(self.F1_score(belta))

    def Dice(self):
        dice = 2 * self.__TP / (self.__RealN + self.__RealP + 1e-5)
        return dice

    def Mean_Dice(self):
        dice = self.Dice()
        return np.nanmean(dice)

    def Recall(self):  # 预测为正确的像素中确认为正确像素的个数
        #TP/ TP+FN
        recall = self.__TP / (self.__RealN + 1e-5)
        return recall
    
    def Mean_Recall(self):
        return np.nanmean(self.Recall())

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.__num_class)
        label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.__num_class ** 2)
        confusion_matrix = count.reshape(self.__num_class, self.__num_class)
        return confusion_matrix

    def add_batch(self, pred, lab):
        
        pred = np.array(pred.cpu())
        lab = np.array(lab.cpu())

        if len(lab.shape) == 4 and (lab.shape[1] > 1):
            lab = np.argmax(lab, axis=1)

        if len(pred.shape) == 4 and (pred.shape[1] > 1):
            pred = np.argmax(pred, axis=1)   

        gt_image = np.squeeze(lab)
        pre_image = np.squeeze(pred)

        assert (np.max(pre_image) <= self.__num_class)
        assert gt_image.shape == pre_image.shape
        self.__confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def calc(self):
        self.__TP = np.diag(self.__confusion_matrix)
        self.__RealN = np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = np.sum(self.__confusion_matrix)

    def reset(self):
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)
        self.__TP = 0.     #TP
        self.__RealN = 0.  # TP+FN
        self.__RealP = 0.  # TP+FP
        self.__sum = 0.  # np.sum(self.__confusion_matrix)






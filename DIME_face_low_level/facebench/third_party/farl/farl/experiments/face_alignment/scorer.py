from typing import Mapping, List, Union, Optional
from collections import defaultdict

import math

import numpy as np
from scipy.integrate import simps
import torch
import torch.distributed as dist

from blueprint.ml import Scorer, all_gather_by_part


class NormalizeInfo:
    def get_unit_dist(self, data) -> float:
        raise NotImplementedError()


class NormalizeByLandmarks(NormalizeInfo):
    def __init__(self, landmark_tag: str, left_id: Union[int, List[int]], right_id: Union[int, List[int]]):
        self.landmark_tag = landmark_tag
        if isinstance(left_id, int):
            left_id = [left_id]
        if isinstance(right_id, int):
            right_id = [right_id]
        self.left_id, self.right_id = left_id, right_id

    def get_unit_dist(self, data) -> float:
        landmark = data[self.landmark_tag]
        unit_dist = np.linalg.norm(landmark[self.left_id, :].mean(0) -
                                   landmark[self.right_id, :].mean(0), axis=-1)
        return unit_dist


class NormalizeByBox(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        return math.sqrt(h * w)


class NormalizeByBoxDiag(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        diag = math.sqrt(w * w + h * h)
        return diag


class NME(Scorer):


    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_infos: Mapping[str, NormalizeInfo]) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_infos = normalize_infos

    def init_evaluation(self):
        self.nmes_sum = defaultdict(float)
        self.count = defaultdict(int)

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')

        for norm_name, norm_info in self.normalize_infos.items():

            unit_dist = norm_info.get_unit_dist(data)


            nme = (np.linalg.norm(
                landmark - pred_landmark, axis=-1) / unit_dist).mean()
            self.nmes_sum[norm_name] += nme

            self.count[norm_name] += 1

    def finalize_evaluation(self) -> Mapping[str, float]:

        names_array: List[str] = list(self.nmes_sum.keys())

        nmes_sum = torch.tensor(
            [self.nmes_sum[name] for name in names_array],
            dtype=torch.float32, device='cuda')
        if dist.is_initialized():
            dist.all_reduce(nmes_sum)

        count_sum = torch.tensor(
            [self.count[name] for name in names_array],
            dtype=torch.int64, device='cuda')
        if dist.is_initialized():
            dist.all_reduce(count_sum)

        scores = dict()


        for name, nmes_sum_val, count_val in zip(names_array, nmes_sum, count_sum):
            scores[name] = nmes_sum_val.item() / count_val.item()


        return scores


class AUC_FR(Scorer):


    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_info: NormalizeInfo,
                 threshold: float, suffix_name: str, step: float = 0.0001,
                 gather_part_size: Optional[int] = 5) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_info = normalize_info
        self.threshold = threshold
        self.suffix_name = suffix_name
        self.step = step
        self.gather_part_size = gather_part_size

    def init_evaluation(self):
        self.nmes = []

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')


        unit_dist = self.normalize_info.get_unit_dist(data)


        nme = (np.linalg.norm(
            landmark - pred_landmark, axis=-1) / unit_dist).mean()
        self.nmes.append(nme)

    def finalize_evaluation(self) -> Mapping[str, float]:


        if dist.is_initialized():
            nmes = all_gather_by_part(self.nmes, self.gather_part_size)
        else:
            nmes = self.nmes
        nmes = torch.tensor(nmes)

        nmes = nmes.sort(dim=0).values.cpu().numpy()


        count = len(nmes)
        xaxis = list(np.arange(0., self.threshold + self.step, self.step))
        ced = [float(np.count_nonzero([nmes <= x])) / count for x in xaxis]
        auc = simps(ced, x=xaxis) / self.threshold
        fr = 1. - ced[-1]

        return {f'auc_{self.suffix_name}': auc, f'fr_{self.suffix_name}': fr}

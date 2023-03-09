from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

classes = ('cloudy', 'uncertain clear', 'probably clear', 'confident clear')
palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 0]]

@DATASETS.register_module()
class MODIS_Dataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpeg', seg_map_suffix='.jpeg', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
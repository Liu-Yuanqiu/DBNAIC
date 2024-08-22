from .field import *
from .dataset import COCO_KD, COCO
from torch.utils.data import DataLoader as TorchDataLoader


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)


def build_image_field(image_field, features_path, max_detections=None):
    if image_field == 'ImageAllFieldWithMask':
        return ImageAllFieldWithMask(detections_path=features_path,
                             max_detections=max_detections,
                             load_in_tmp=False)
    elif image_field == 'ImageSwinRegionGridWithMask':
        return ImageSwinRegionGridWithMask(detections_path=features_path,
                             max_detections=max_detections,
                             load_in_tmp=False)
    elif image_field == 'ImageSwinRegionWithMask':
        return ImageSwinRegionWithMask(detections_path=features_path,
                             max_detections=max_detections,
                             load_in_tmp=False)
    elif image_field == 'ImageSwinGridWithMask':
        return ImageSwinGridWithMask(detections_path=features_path,
                             max_detections=max_detections,
                             load_in_tmp=False)
    elif image_field == 'ImageTransform':
        return ImageTransform(detections_path=features_path,
                             max_detections=max_detections,
                             load_in_tmp=False)
    else:
        raise NotImplementedError('No field: {}'.format(image_field))

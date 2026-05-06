from .base import BaseConverter, FathomNetRecord
from .coco import COCOConverter
from .pascal_voc import PascalVOCConverter
from .yolo import YOLOConverter
from .folder_encoded import FolderEncodedConverter
from .fishclef import FishCLEFConverter

CONVERTER_REGISTRY = {
    "coco_json": COCOConverter,
    "pascal_voc": PascalVOCConverter,
    "yolo": YOLOConverter,
    "folder_encoded": FolderEncodedConverter,
    "fishclef_xml": FishCLEFConverter,
}

import re
from pathlib import Path
from typing import Any, List, Tuple, Dict

from PIL import Image
from torch.utils.data import Dataset

from game_model import BallStateType
from .model import LABEL_INDEX_MAP


class PoolBallDataset(Dataset):
    path: Path
    transform: Any
    data: List[Tuple[Any, int]]

    def __init__(self, path: Path, transform):
        self.path = path
        self.transform = transform

        self.data = [
            (
                PoolBallDataset._read_img(path, transform=transform),
                LABEL_INDEX_MAP[PoolBallDataset._read_label_from_path(path)]
            )
            for path in PoolBallDataset._list_img_files(path)
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def _read_img(path: Path, transform: Any = None):
        image = Image.open(path)
        if transform:
            image = transform(image)
        return image

    @staticmethod
    def _read_label_from_path(path: Path) -> BallStateType:
        match_obj = re.match(r'^id_\d+_\d+_([a-z]+)(?:\d+)?.PNG$', path.name, flags=re.IGNORECASE)
        if match_obj is None:
            raise ValueError('invalid file name')

        value = match_obj.group(1)

        translation_dict: Dict[str, BallStateType] = {
            'solid': 'solid',
            'solids': 'solid',
            'whole': 'solid',
            'full': 'solid',
            'stripes': 'stripes',
            'striped': 'stripes',
            'stripe': 'stripes',
            'eight': 'eight',
            'black': 'eight',
            'cue': 'cue',
            'white': 'cue',
        }

        return translation_dict[value]

    # @staticmethod
    # def _read_label_csv(dataset_path: Path) -> Dict[int, str]:
    #     label_file_path = dataset_path / 'id_mapping.txt'
    #     with label_file_path.open(mode='rt', encoding='ascii', newline='\n') as file:
    #         body_lines = list(line.strip() for line in file)[1:]  # skip header line
    #         return {
    #             int(id_str): label
    #             for id_str, label in map(lambda line: line.split(','), body_lines)
    #         }

    @staticmethod
    def _list_img_files(dataset_path: Path) -> List[Path]:
        # def extract_id(path: Path) -> int:
        #     match_obj = re.match(r'^id_(\d+).PNG$', path.name, flags=re.IGNORECASE)
        #     assert match_obj is not None
        #     return int(match_obj.group(1))

        def is_png_file(path: Path) -> bool:
            return path.name.lower().endswith('.png')

        return sorted([
            path
            for path in dataset_path.iterdir()
            if is_png_file(path)
        ])

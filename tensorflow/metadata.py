

import json
import os
import pandas as pd
from config import Constant



C = Constant()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]




class Metadata(metaclass=Singleton):
    def __init__(self):
        self.dataset_dir = C.DATASET_DIR
        self._metadata = self._load_metadata_dataframe()
        self._norm_metadata = self._load_norm_metadata()

    @property
    def metadata(self):
        return self._metadata

    @property
    def norm_metadata(self):
        return self._norm_metadata

    def _load_norm_metadata(self):
        return pd.read_json(os.path.join(self.dataset_dir, "NORM_S2_patch.json")).to_dict()

    def _load_metadata_dataframe(self):
        """
        Processing patch metadata and extracting the following features:
        Patch_path, Fold, Patch_id, N_parcel, Parcel_cover, Tile, i_date (date of time series object {i})
        :return: pandas dataframe with metadata of Sentinel-2 patches.
        """
        with open(os.path.join(self.dataset_dir, 'metadata.geojson')) as f:
            metadata = json.load(f)
            metadata = pd.json_normalize(metadata, record_path='features', max_level=1)

        # Processing dates
        dates = pd.DataFrame(metadata['properties.dates-S2'].values.tolist())
        for column in dates.columns:
            d = dates[column]
            dates[f'{column}_date'] = pd.to_datetime(d, format='%Y%m%d')
            dates.drop(columns=[column], inplace=True)

        metadata['Semantic_segmentation_path'] = metadata['properties.ID_PATCH'].apply(lambda x: f'{C.DATASET_DIR}ANNOTATIONS/TARGET_{x}.npy')
        metadata['Patch_path'] = metadata['properties.ID_PATCH'].apply(lambda x: f'{C.DATASET_DIR}DATA_S2/S2_{x}.npy')
        metadata.drop(
            columns=['id', 'type', 'properties.id', 'geometry.type', 'geometry.coordinates', 'properties.dates-S2'],
            inplace=True)
        metadata.rename(
            columns={'properties.Fold': 'Fold', 'properties.N_Parcel': 'N_parcel', 'properties.ID_PATCH': 'Patch_id',
                     'properties.Parcel_Cover': 'Parcel_cover', 'properties.TILE': 'Tile'}, inplace=True)

        patches_metadata = pd.concat([metadata, dates], ignore_index=False, axis=1)
        patches_metadata.set_index('Patch_path', inplace=True)

        return patches_metadata
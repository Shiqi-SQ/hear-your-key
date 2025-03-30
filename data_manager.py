import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
class DataManager:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.samples_file = self.data_dir / 'key_samples.json'
        self.features_dir = self.data_dir / 'features'
        self.features_dir.mkdir(exist_ok=True)
        self.samples_data = self._load_samples()
    def _load_samples(self) -> pd.DataFrame:
        if self.samples_file.exists():
            return pd.read_json(self.samples_file)
        return pd.DataFrame(columns=['key', 'feature_file', 'timestamp'])
    def save_sample(self, key: str, features: Dict[str, Any]) -> bool:
        try:
            feature_file = f'{key}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.npy'
            feature_path = self.features_dir / feature_file
            np.save(feature_path, features)
            new_sample = pd.DataFrame([
                {
                    'key': key,
                    'feature_file': str(feature_file),
                    'timestamp': pd.Timestamp.now()
                }
            ])
            self.samples_data = pd.concat([self.samples_data, new_sample], ignore_index=True)
            self.samples_data.to_json(self.samples_file)
            return True
        except Exception as e:
            print(f'保存样本时出错: {e}')
            return False
    def get_samples_for_key(self, key: str) -> list:
        key_samples = self.samples_data[self.samples_data['key'] == key]
        features_list = []
        for _, sample in key_samples.iterrows():
            feature_path = self.features_dir / sample['feature_file']
            if feature_path.exists():
                features = np.load(feature_path, allow_pickle=True).item()
                features_list.append(features)
        return features_list
    def get_all_keys(self) -> list:
        return self.samples_data['key'].unique().tolist()
    def get_sample_count(self, key: Optional[str] = None) -> int:
        if key is not None:
            return len(self.samples_data[self.samples_data['key'] == key])
        return len(self.samples_data)
import os
from huggingface_hub import snapshot_download
from shared import ensure_dir_exists


class HFModelDownloader:
    def __init__(self, model_id, token=None):
        self.model_id = model_id
        self.token = token

    def get_model_dir(self, cwd):
        return os.path.join(cwd, "models", self.model_id)

    def download_model(self, cwd):
        model_dir = self.get_model_dir(cwd)
        ensure_dir_exists(model_dir)
        snapshot_download(self.model_id, local_dir=model_dir, token=self.token)
        return model_dir

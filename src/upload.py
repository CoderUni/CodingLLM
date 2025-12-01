import os
from src.utils.config_loader import config


from modelscope.hub.api import HubApi
from huggingface_hub import HfApi


# HuggingFace config
# LOGIN TO HUGGINGFACE
# See https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication
hfAPI = HfApi()

# Modelscope config
MODELSCOPE_ACCESS_TOKEN = os.getenv("MODELSCOPE_ACCESS_TOKEN")
api = HubApi()
api.login(MODELSCOPE_ACCESS_TOKEN)

# Upload config
REPO_ID = config["save"]["repo_id"]

# Set this to the correct model path
MODEL_PATH = "/mnt/storage/metnet/coding_llm/final_finetuned_model/"

# If the upload fails, try using git lfs to upload instead.
if __name__ == "__main__":
    # Upload to HuggingFace Hub
    hfAPI.upload_folder(
        folder_path=MODEL_PATH,
        path_in_repo=".",  # Upload to a specific folder
        repo_id=REPO_ID,
        repo_type="model",
        ignore_patterns="**/logs/*.txt",  # Ignore all text logs
    )

    # Upload to Modelscope Hub
    api.push_model(
        model_id=REPO_ID,
        model_dir=MODEL_PATH,
    )

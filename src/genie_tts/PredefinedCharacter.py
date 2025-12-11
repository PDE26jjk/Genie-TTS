from huggingface_hub import snapshot_download
import os


def download_chara(chara: str, version: str = "v2ProPlus"):
    local_dir = os.path.join("CharacterModels", version, chara)
    if os.path.exists(local_dir):
        print(f"✔ Model for '{chara}' already exists locally. Skipping download.")
        return local_dir

    print(f"🚀 Starting download of model for character '{chara}'. This may take a few moments... ⏳")
    remote_path = f"CharacterModels/{version}/{chara}/*"
    local_dir = snapshot_download(
        repo_id="High-Logic/Genie",
        repo_type="model",
        allow_patterns=remote_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 软链接
    )
    print(f"🎉 All model files for '{chara}' have been downloaded to '{os.path.abspath(local_dir)}' 📂")
    return local_dir

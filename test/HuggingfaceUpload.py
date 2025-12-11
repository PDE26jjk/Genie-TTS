from huggingface_hub import upload_folder
import os

r"""
upload_folder(
    folder_path=r"C:\Users\Haruka\Desktop\GENIE\GENIE_CPU_RUNTIME_OLD\GenieData",  # 本地文件夹
    path_in_repo="GenieData",
    repo_id="High-Logic/Genie",
    repo_type="model",
    commit_message="Upload folder via Python"
)
"""


def upload_chara(
        chara: str = 'feibi',
        version: str = 'v2ProPlus',  # v2ProPlus v2
):
    midori_dir = r'C:\Users\Haruka\Desktop\Midori\Data\character_resource'
    source_dir = os.path.join(midori_dir, chara, 'tts')

    upload_folder(
        folder_path=source_dir,  # 本地文件夹
        path_in_repo=f"CharacterModels/{version}/{chara}",
        repo_id="High-Logic/Genie",
        repo_type="model",
        commit_message="Upload folder via Python"
    )


upload_chara('thirtyseven')

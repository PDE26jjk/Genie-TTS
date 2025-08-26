import genie_tts as genie

# 目前 convert_to_onnx 只能转换 v2 的模型。
genie.convert_to_onnx(
    torch_pth_path=r"C:\Users\Haruka\Desktop\Midori\Dockers\mnt\Midori\Model\irene\irene.pth",
    torch_ckpt_path=r"C:\Users\Haruka\Desktop\Midori\Dockers\mnt\Midori\Model\irene\irene.ckpt",
    output_dir='./Output'
)

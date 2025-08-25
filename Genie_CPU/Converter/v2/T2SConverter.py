import torch
import onnx
import numpy as np
import json
import os
from collections import OrderedDict

from ..load_state_dict import load_gpt_model


class T2SModelConverter:
    """
    一个专门的转换器，用于处理 t2s (Text-to-Speech) 模型。
    - PyTorch 模型: .ckpt 文件
    - ONNX 模型: t2s_stage_decoder_fp32.onnx
    - 遵循特定的键名映射规则。
    """

    def __init__(self,
                 torch_ckpt_path: str,
                 stage_decoder_onnx_path: str,
                 first_stage_decoder_onnx_path: str,
                 key_list_file: str,
                 output_dir: str,
                 cache_dir: str,
                 ):
        self.torch_ckpt_path: str = torch_ckpt_path
        self.stage_decoder_onnx_path: str = stage_decoder_onnx_path
        self.first_stage_decoder_onnx_path: str = first_stage_decoder_onnx_path
        self.key_list_file: str = key_list_file
        self.output_dir: str = output_dir
        self.cache_dir: str = cache_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 定义输出文件路径
        self.fp16_bin_path: str = os.path.join(self.output_dir, "t2s_shared_fp16.bin")
        self.index_table_path: str = os.path.join(self.cache_dir, "t2s_weights_index_fp32.json")
        self.relinked_encoder_path: str = os.path.join(self.output_dir, "t2s_encoder_fp32.onnx")
        self.relinked_stage_decoder_path: str = os.path.join(self.output_dir, "t2s_stage_decoder_fp32.onnx")
        self.relinked_first_stage_decoder_path: str = os.path.join(self.output_dir, "t2s_first_stage_decoder_fp32.onnx")
        self.reconstructed_fp32_bin_path = os.path.join(self.output_dir, "t2s_shared_fp32.bin")

    def step1_create_fp16_bin_with_key_mapping(self):
        """
        (1) 根据特定的键映射规则，从 .ckpt 创建 fp16 .bin 和 fp32 索引。
            (已根据用户验证脚本的正确逻辑进行最终修正)
        """
        if not os.path.exists(self.key_list_file):
            raise FileNotFoundError(f"错误: 阶段 1 需要 Key 列表文件，但未找到: {self.key_list_file}")

        try:
            with open(self.key_list_file, 'r') as f:
                onnx_keys = [line.strip() for line in f.readlines()]

            ckpt_data = load_gpt_model(self.torch_ckpt_path)
            if 'weight' not in ckpt_data:
                raise KeyError(f"❌ 错误: 在 .ckpt 文件中找不到 'weight' 键。文件顶层键为: {list(ckpt_data.keys())}")

            torch_state_dict = ckpt_data['weight']

            index_table = OrderedDict()
            current_fp32_offset = 0

            with open(self.fp16_bin_path, 'wb') as f_bin:
                for onnx_key in onnx_keys:
                    # --- 应用正确的键映射逻辑 ---
                    # 1. 对 ONNX 键应用其自身的转换规则
                    transformed_onnx_key = onnx_key.replace('transformer_encoder', 'h')
                    # 2. 构造在原始 PyTorch 字典中要查找的键：
                    #    即对转换后的 ONNX 键，应用 PyTorch 规则的“逆操作”（添加 'model.' 前缀）
                    torch_lookup_key = f"model.{transformed_onnx_key}"
                    # 3. 在 PyTorch state_dict 中查找
                    torch_tensor = torch_state_dict.get(torch_lookup_key)
                    # 写入 fp16 数据
                    numpy_array_fp16 = torch_tensor.to(torch.float16).cpu().numpy()
                    f_bin.write(numpy_array_fp16.tobytes())
                    # 记录 fp32 布局
                    tensor_length_fp32 = numpy_array_fp16.nbytes * 2
                    index_table[onnx_key] = {'offset': current_fp32_offset, 'length': tensor_length_fp32}
                    current_fp32_offset += tensor_length_fp32

            with open(self.index_table_path, 'w') as f_json:
                json.dump(index_table, f_json, indent=4)  # type: ignore
        except Exception as e:
            print(f"❌ 阶段 1 失败: {e}")
            raise

    def step2_relink_onnx_for_fp32(self, old_model: str, new_model: str):
        """
        (2) 根据 fp32 索引表，修改 ONNX 模型，使其链接到未来的全精度 .bin。
            (使用与第一个脚本相同的、更稳定的底层方法)
        """
        if not os.path.exists(self.index_table_path):
            raise FileNotFoundError(f"错误: 阶段 2 需要索引文件，但未找到: {self.index_table_path}")

        try:
            # 加载描述 fp32 布局的索引表
            with open(self.index_table_path, 'r') as f:
                index_table = json.load(f)

            # 加载 ONNX 模型结构 (使用 self.onnx_model_path)
            model = onnx.load_model(old_model, load_external_data=False)

            # 这个 ONNX 模型将要链接的 .bin 文件名
            reconstructed_bin_filename = os.path.basename(self.reconstructed_fp32_bin_path)

            for tensor in model.graph.initializer:
                if tensor.name in index_table:
                    # 清除可能存在的原始数据
                    tensor.ClearField('raw_data')
                    # 设置数据存储位置为外部
                    tensor.data_location = onnx.TensorProto.EXTERNAL
                    info = index_table[tensor.name]
                    # 清空旧的外部数据链接
                    del tensor.external_data[:]
                    # 设置新的链接信息
                    keys = ["location", "offset", "length"]
                    values = [reconstructed_bin_filename, str(info['offset']), str(info['length'])]

                    for k, v in zip(keys, values):
                        entry = tensor.external_data.add()
                        entry.key = k
                        entry.value = v

            # 保存修改后的、链接到 fp32 权重的 ONNX 模型
            onnx.save(model, new_model)

        except Exception as e:
            print(f"❌ 阶段 2 失败: {e}")
            raise

    @staticmethod
    def step3_reconstruct_fp32_bin_from_fp16(fp16_bin_path: str, output_fp32_bin_path: str):
        """
        (3) 静态工具函数：从半精度 .bin 文件还原出全精度 .bin 文件。
        """
        fp16_array = np.fromfile(fp16_bin_path, dtype=np.float16)
        fp32_array = fp16_array.astype(np.float32)
        fp32_array.tofile(output_fp32_bin_path)
        print(f"✅ 还原成功！")

    def run_full_process(self):
        self.step1_create_fp16_bin_with_key_mapping()
        self.step2_relink_onnx_for_fp32(self.stage_decoder_onnx_path, self.relinked_stage_decoder_path)
        self.step2_relink_onnx_for_fp32(self.first_stage_decoder_onnx_path, self.relinked_first_stage_decoder_path)
        # print("🎉🎉🎉 T2S 模型转换全流程已成功完成！ 🎉🎉🎉")

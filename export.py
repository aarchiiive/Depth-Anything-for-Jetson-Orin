import argparse

import time

import os
from pathlib import Path

import torch
import tensorrt as trt
from depth_anything import DepthAnything


def export(
    weights_path: str,
    save_path: str,
    input_size: int,
    onnx: bool = True,
):
    """
    weights_path: str -> Path to the PyTorch model(local / hub)
    save_path: str -> Directory to save the model
    input_size: int -> Width and height of the input image(e.g. 308, 364, 406, 518)
    onnx: bool -> Export the model to ONNX format
    """
    weights_path = Path(weights_path)

    os.makedirs(save_path, exist_ok=True)

    # Load the model
    model = DepthAnything.from_pretrained(weights_path).to('cpu').eval()

    # Create a dummy input
    dummy_input = torch.ones((3, input_size, input_size)).unsqueeze(0)
    _ = model(dummy_input)
    onnx_path = Path(save_path) / f"{weights_path.stem}_{input_size}.onnx"

    # Step 1: Export the model to ONNX format
    # Hint: Use `torch.onnx.export`
    # Reference: https://pytorch.org/docs/stable/onnx.html
    ### TODO: Write your code here (1 line)

    ###############################################################################################################################################

    time.sleep(2)

    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    # Step 2: Convert the ONNX model to TensorRT
    # Hint: Initialize `network` & `parser`
    # Reference:
    # - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder.create_network
    # - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/parsers/Onnx/pyOnnx.html
    ### TODO: Write your code here (2 lines)


    ###############################################################################################################################################

    # Step 3: Set up the builder config
    # Required parameters:
    # - flag: FP32 | FP16 (Choose your precision)
    # - workspace: 2 << 30
    # Hint: Initialize `config`
    # Reference:
    # - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder.create_builder_config
    # - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html?highlight=builderflag#tensorrt.BuilderFlag
    # - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html?highlight=builderflag#tensorrt.MemoryPoolType
    ### TODO: Write your code here (3 lines)



    ###############################################################################################################################################

    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')

    # Step 4: Build and save the TensorRT engine
    # Hint: Use `builder` to build the engine
    # Reference: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder.build_serialized_network
    ### TODO: Write your code here (1 line)

    ###############################################################################################################################################

    with open(onnx_path.with_suffix(".trt"), "wb") as f:
        f.write(serialized_engine)

if __name__ == '__main__':
    export(
        weights_path="LiheYoung/depth_anything_vits14", # local hub or online
        save_path="weights", # folder name
        input_size=308, # 308 | 364 | 406 | 518
        onnx=True,
    )
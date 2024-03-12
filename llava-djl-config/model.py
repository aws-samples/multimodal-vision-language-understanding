import hashlib
import math
import numpy as np
import os,io
import torch
from PIL import Image
import base64
from io import BytesIO
import json

from dataclasses import dataclass

from transformers import AutoProcessor, LlavaForConditionalGeneration

from typing import List, Optional
from djl_python import Input, Output



@dataclass 
class Config:
    # models can optionally be passed in directly
    caption_model = None
    caption_processor = None
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")

   
class Llava():
    def __init__(self, config: Config, properties):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.load_llava_model(properties)

    def load_llava_model(self, properties):
        if self.config.caption_model is None:
            model_path = properties["model_id"]
            if any(os.listdir(model_path)):
                files_in_folder = os.listdir(model_path)
                print('model path files:')
                for file in files_in_folder:
                    print(file)
            else:
                raise ValueError('Please make sure the model artifacts are uploaded to s3')

            print(f'model path: {model_path}')
            self.caption_processor = AutoProcessor.from_pretrained(model_path)
            print(self.caption_processor)
            model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype, device_map=torch.device(self.device), cache_dir="/tmp",)
            model.eval()
            #model = model.to(self.config.device)
            self.caption_model = model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    def generate_caption(self, pil_image: Image, prompt: Optional[str]=None, params: Optional[dict]={}) -> str:
        assert self.caption_model is not None, "No caption model loaded."

        inputs = self.caption_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        inputs = inputs.to(self.dtype)

        with torch.no_grad():
            generate_ids = self.caption_model.generate(**inputs, **params)

        return self.caption_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()



config = None
_service = None

def handle(inputs: Input) -> Optional[Output]:
    global config, _service
    if not _service:
        config = Config()
        _service = Llava(config, inputs.get_properties())
    
    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    if 'image' in data:
        base64_image_string = data.pop("image")
        f = BytesIO(base64.b64decode(base64_image_string))
        input_image = Image.open(f).convert("RGB")
    else:
        input_image = None
    
    if 'prompt' in data:
        prompt = data.pop("prompt")
    else:
        prompt = None
    
    params = data["parameters"] if 'parameters' in data else {}
    print(params)
    
    generated_text = _service.generate_caption(input_image, prompt, params)
    print(generated_text)
    return Output().add(generated_text)

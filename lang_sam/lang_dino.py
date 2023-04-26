import os
from urllib import request
from PIL import Image

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download


CACHE_PATH = os.environ.get("TORCH_HOME", "~/.cache/torch/hub/checkpoints")


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_groundingdino():
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    
    return groundingdino

def predict_dino(frame, text_prompt, box_threshold, text_threshold):
    PIL_image = Image.fromarray(frame)
    image_trans = transform_image(PIL_image)
    #tensor_frame = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
    boxes, logits, phrases = predict(model=build_groundingdino(),
                                        image=image_trans,
                                        caption=text_prompt,
                                        box_threshold=box_threshold,
                                        text_threshold=text_threshold,
                                        device=device)
    W, H = PIL_image.size
    boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    return boxes, logits, phrases

    # def predict_sam(self, image_pil, boxes):
    #     image_array = np.asarray(image_pil)
    #     self.sam.set_image(image_array)
    #     transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    #     masks, _, _ = self.sam.predict_torch(
    #         point_coords=None,
    #         point_labels=None,
    #         boxes=transformed_boxes.to(self.sam.device),
    #         multimask_output=False,
    #     )
    #     return masks.cpu()

    # def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
    #     boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
    #     masks = torch.tensor([])
    #     # if len(boxes) > 0:
    #     #     masks = self.predict_sam(image_pil, boxes)
    #     #     masks = masks.squeeze(1)
    #     return masks, boxes, phrases, logits

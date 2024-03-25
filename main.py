import cv2
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="/home/tom/Models/SAM/sam_vit_h_4b8939.pth")
sam.to('cuda')
sam = sam.to(torch.float16)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam, output_mode='uncompressed_rle')

image = cv2.imread("/home/tom/Junk/small.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with torch.inference_mode():
    mask_generator.generate(image)
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            tic = time.time()
            mask_generator.generate(image)
            print(time.time() - tic)
prof.export_chrome_trace("trace.json")
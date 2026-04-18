import torch
from utils.ddpm import Diffusion
import matplotlib.pyplot as plt
from utils.utils import set_seed

set_seed(0)
from guided_diffusion.script_util import create_model

path = "weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar"


model = create_model(image_size=32, num_channels=64, num_res_blocks=3, input_channels=3).to("cuda")
diff = Diffusion(device="cuda", img_size=32, noise_steps=1000, schedule_name="cosine", channels=3)
checkpoint = torch.load(path, map_location="cuda")
model.load_state_dict(checkpoint["model_state"])
model.eval()

x = diff.sample(model, n=1)

x = (x - x.min()) / (x.max() - x.min())

print(x.shape)

plt.figure(figsize=(6, 6))
plt.imshow(x[0].permute(1, 2, 0).cpu().detach().numpy())
plt.axis("off")
plt.colorbar()
plt.show()

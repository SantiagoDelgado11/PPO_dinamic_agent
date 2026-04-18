import os


batch_size = 170
image_size = 32
seed = 0
gpu_id = 0
save_image = "False"
plot = "False"
idx_list = [i for i in range(0, batch_size)]
wandb_id = "b879bf20f3c31bfcf13289e363f4d3394f7d7671"
use_wandb = "True"
device = "cuda"
weights = "weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar"  # Diffusion weights

sampling_ratio = 1 / 12
sampling_method = "uniform"
algo = "DDNM"


# DDNM parameters
ddnm_eta = 1.0

# DPS parameters
dps_scale = 0.0125

# DiffPIR parameters
CG_iters_diffpir = 100
noise_level_img = 0.0


project_name = "STSIVA_2026_pruebas_de_codigo"

name = f"lote_completo_{algo}_{sampling_method}_{int(180 * sampling_ratio)}"

print(f"Iniciando reconstrucción SPC para {batch_size} imágenes con {algo}...")

comando = (
    f"python run_one_img.py "
    f"--weights {weights} "
    f"--batch_size {batch_size} "
    f"--image_size {image_size} "
    f"--seed {seed} "
    f"--device {device} "
    f"--gpu_id {gpu_id} "
    f"--save_image {save_image} "
    f"--plot {plot} "
    f"--sampling_ratio {sampling_ratio} "
    f"--sampling_method {sampling_method} "
    f"--algo {algo} "
    f"--dps_scale {dps_scale} "
    f"--CG_iters_diffpir {CG_iters_diffpir} "
    f"--noise_level_img {noise_level_img} "
    f"--use_wandb {use_wandb} "
    f"--wandb_id {wandb_id} "
    f"--project_name {project_name} "
    f"--name {name}"
)

os.system(comando)
print("¡Reconstrucción finalizada!")

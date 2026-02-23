import numpy as np
import os
from render import MitsubaRenderer
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Physical Parameters from your notes ---
C_EFF = 8.73e5  # Efficiency constant
MAX_ELECTRONS = 10600  # Targeted max electron well capacity
B_TAU = 4.72  #
READ_NOISE_STD = 5.29  # Read noise standard deviation (STD)
K_QUANT = 8.93  # Quantization constant

# --- File Paths ---
csv_file = '/home/danino/PycharmProjects/pythonProject/data/Dor_2satellites_overpass.csv'
cloud_data_file = '/wdata_visl/danino/dataset_128x128x200_overlap_64_stride_7x7_split(beta,U,V,W)/0000005920/sample_012.pkl'
output_vol_file = 'temp/my_cloud.vol'

os.makedirs('temp', exist_ok=True)

# --- Render Settings ---
SPP = 512
NUM_RUNS = 10
SEED_BASE = 42

renderer_params = {
    'overpass_csv': csv_file,
    'overpass_indices': [9],
    'spp': SPP,
    'g_value': 0,
    'cloud_width': 128,
    'image_res': 256,
    'fov': 0.25,
    'voxel_res': 0.02,
    'scene_scale': 1000.0,
    'cloud_zrange': [0.0, 4.0],
    'satellites': 1,
    'timestamps': 1,
    'pad_image': False,
    'dynamic_emitter': True,
    'centralize_cloud': True,
    'bitmaps_required': False,
    'vol_path': output_vol_file
}

image_stack_radiance = []

print(f"Starting rendering with SPP: {SPP}...")

for run_idx in tqdm(range(NUM_RUNS)):
    renderer_params_with_seed = dict(renderer_params)
    renderer_params_with_seed['seed'] = SEED_BASE + run_idx

    renderer = MitsubaRenderer(**renderer_params_with_seed)
    renderer.read_overpass_csv()
    renderer.camera_params()
    renderer.create_sensors()
    renderer.write_vol_file(sample_path=cloud_data_file, vol_path=output_vol_file)
    renderer.set_scenes()

    tensor_stacks, _ = renderer.render_scenes()
    img = tensor_stacks[0][0]
    image_stack_radiance.append(img)

# 1. Stack and calculate Delta_t based on the maximum radiance found across all runs
image_stack_radiance = np.stack(image_stack_radiance, axis=0)  # (Runs, H, W)
max_radiance = np.max(image_stack_radiance)

# Delta_t = Max_Electrons / (Max_Radiance * C_eff)
delta_t = MAX_ELECTRONS / (max_radiance * C_EFF)  #
delta_t_ms = delta_t * 1000
print(f"Calculated Delta_t for max {MAX_ELECTRONS}e-: {delta_t:.6f} seconds ({delta_t_ms:.3f} ms)")

# 2. Convert Radiance stack to Electrons (u_e)
# u_e = Radiance * C_eff * Delta_t
conversion_factor = C_EFF * delta_t
image_stack_electrons = image_stack_radiance * conversion_factor  #

# 3. Calculate Mean and Variance in Electron units
valid_mask = np.mean(image_stack_electrons, axis=0) > 0
mean_ue = np.mean(image_stack_electrons, axis=0)
mc_variance_ue = np.var(image_stack_electrons, axis=0)

# Filter data for plotting (valid pixels only, mean > 1 electron)
flat_mean = mean_ue[valid_mask].flatten()
flat_mc_var = mc_variance_ue[valid_mask].flatten()

mask_above_1 = flat_mean > 1
flat_mean = flat_mean[mask_above_1]
flat_mc_var = flat_mc_var[mask_above_1]

# 4. Calculate Poisson Noise Model Variance
# Variance_poisson = mean_ue (Shot Noise)
poisson_var = flat_mean

# 5. Calculate Full Camera Noise Model Variance
# sigma_sum^2 = (shot_noise) + (dark_current) + (read_noise) + (quantization)
# Note: shot_noise is already mean_ue in electrons
camera_var = flat_mean + (B_TAU * delta_t) + READ_NOISE_STD + K_QUANT

# --- Visualization ---
plt.figure(figsize=(12, 7))

# Sort indices by mean for a cleaner line plot or use a scatter plot
sort_idx = np.argsort(flat_mean)
sorted_mean = flat_mean[sort_idx]
sorted_mc_var = flat_mc_var[sort_idx]

plt.scatter(flat_mean, flat_mc_var, alpha=0.3, s=1, label='Monte Carlo Variance (Render)', color='gray')
plt.plot(sorted_mean, sorted_mean, 'r--', linewidth=2, label='Poisson Noise Model (Var = Mean)')
plt.plot(sorted_mean, camera_var[sort_idx], 'b--', linewidth=2, label='Full Camera Noise Model')

plt.title(f'Variance Comparison: Monte Carlo vs. Camera Noise (SPP={SPP})')
plt.xlabel('Mean Signal (Electrons - $u_e$)')
plt.ylabel('Variance ($\sigma^2$)')
plt.ylim(top=0.80000)
#plt.yscale('log')  # Log scale helps see the MC noise clearly
#plt.xscale('log')
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()
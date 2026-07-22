import pickle
import numpy as np

pkl_path = 'data/output/satellite_output_steps/step_0000.pkl'
with open(pkl_path, 'rb') as f:
    obj = pickle.load(f)

print('Type:', type(obj))
if hasattr(obj, 'keys'):
    print('Keys:', list(obj.keys()))
    for k, v in obj.items():
        print(f'Key: {k}, Type: {type(v)}, Shape: {getattr(v, "shape", None)}')
else:
    print('No keys')
    print('Shape:', getattr(obj, 'shape', None))

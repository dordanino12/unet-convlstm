import os.path

import numpy as np
import scipy
import pandas as pd
from PIL import Image
import struct
import torch
import os
import mitsuba as mi
import pickle

if torch.cuda.is_available():
    mi.set_variant('cuda_ad_rgb')
else:
    mi.set_variant('llvm_ad_rgb')


# mi.set_variant('llvm_ad_rgb')
# from mitsuba import ScalarTransform4f


class MitsubaRenderer:
    def __init__(self, overpass_csv, overpass_indices, spp, g_value=0, cloud_width=128, image_res=256, fov=0.25, voxel_res=0.02, scene_scale=1e3,
                 cloud_zrange=[0, 4],
                 satellites=3, timestamps=2, pad_image=True, dynamic_emitter=True, centralize_cloud=True,
                 bitmaps_required=True,
                 vol_path=None):
        self.overpass_csv = overpass_csv
        self.overpass_indices = overpass_indices
        self.spp = spp
        self.g_value = g_value
        self.cloud_width = cloud_width # 128 pixels
        self.voxel_res = voxel_res # 0.02 km per pixel
        self.scene_scale = scene_scale
        self.cloud_zrange = cloud_zrange
        self.satellites = satellites
        self.timestamps = timestamps
        self.pad_image = pad_image
        self.dynamic_emitter = dynamic_emitter
        self.centralize_cloud = centralize_cloud
        self.bitmaps_required = bitmaps_required
        self.vol_path = vol_path

        self.W = self.cloud_width * self.voxel_res # km for all the image
        self.sat_Wx = []
        self.sat_Wy = []
        self.sat_H = []
        self.sat_azimuth = []
        self.sat_zenith = []
        self.sun_azimuth = []
        self.sun_zenith = []

        self.cloud_zcenter = sum(cloud_zrange) / 2

        self.sensors = []
        self.fov = fov
        self.film_dim = image_res

        self.scenes_dict = []
        self.scenes = []
        self.update_required = True

    def read_overpass_csv(self):
        satelite_df = pd.read_csv(self.overpass_csv)
        idx = self.overpass_indices

        sat_coords_lst = [satelite_df.loc[i, "sat ENU coordinates [km]"] for i in idx]
        self.sat_Wx = [float(coords[1:-1].split(',')[0]) for coords in sat_coords_lst]
        self.sat_Wy = [float(coords[1:-1].split(',')[1]) for coords in sat_coords_lst]
        self.sat_H = [float(coords[1:-1].split(',')[2]) for coords in sat_coords_lst]

        self.sat_azimuth = [float(satelite_df.loc[i, "sat azimuth [deg]"]) for i in idx]
        self.sat_zenith = [float(satelite_df.loc[i, "sat zenith [deg]"]) for i in idx]

        if self.dynamic_emitter:
            self.sun_azimuth = [float(satelite_df.loc[self.overpass_indices[i * self.satellites], "sun azimuth [deg]"])
                                for i in range(self.timestamps)]
            self.sun_zenith = [float(satelite_df.loc[self.overpass_indices[i * self.satellites], "sun zenith [deg]"])
                               for i in range(self.timestamps)]
        else:
            self.sun_azimuth = float(satelite_df.loc[self.overpass_indices[0], "sun azimuth [deg]"])
            self.sun_zenith = float(satelite_df.loc[self.overpass_indices[0], "sun zenith [deg]"])

    def camera_params(self):
        limit_idx = np.argmax(self.sat_zenith)
        nadir_idx = np.argmin(self.sat_zenith)

        theta_z = self.sat_zenith[limit_idx]
        H_z = self.sat_H[limit_idx]
        H_0 = self.sat_H[nadir_idx]
        Dz = np.tan(theta_z * (np.pi / 180)) * H_z

        # if self.pad_image:
        #     self.fov = 2 * (-theta_z + np.arctan((Dz + self.W / 2) / (H_z - self.cloud_zrange[1])) * (180 / np.pi))
        #     #self.film_dim = int(
        #     #    np.ceil(2 * (H_z - self.cloud_zrange[1]) * np.tan(self.fov / 2 * np.pi / 180) / self.voxel_res))
        # else:
        #     self.fov = 2 * np.arctan((self.W / 2) / (H_0 - self.cloud_zrange[1])) * (180 / np.pi) # this in degree unit
        #     #self.film_dim = self.cloud_width

    def create_sensors(self):
        # Calculate the cloud's center Z coordinate once
        cloud_target_z = self.cloud_zcenter * 2.5 # 4.22 for 256 , 2.5 for 128 or maybe try 2.2 , for 512 on 512 8.44+-

        for i in range(len(self.overpass_indices)):
            # --- THIS IS THE FIX ---
            # Remove the extra 'sensor_rotation'
            # 1. Set the correct 'target' (the cloud's center)
            # 2. Use a standard 'up' vector (e.g., [0, 1, 0] for Y-up)
            # 3. Use *only* the look_at transform

            sensor_to_world = mi.scalar_rgb.Transform4f.look_at(
                target=[0, 0,cloud_target_z],  # <-- BUG 1 FIX: Point at the cloud center
                origin=[self.sat_Wy[i], self.sat_Wx[i], self.sat_H[i]],
                up=[1, 0, 0]
            )

            self.sensors.append(mi.load_dict({
                'type': 'perspective',
                'fov': self.fov,
                'to_world': sensor_to_world,  # <-- BUG 2 FIX: No extra rotation
                'film': {
                    'type': 'hdrfilm',
                    'width': self.film_dim, 'height': self.film_dim,
                    'filter': {'type': 'tent'}
                }
            }))

    def write_vol_file(self, data=None, sample_path=None, param_type='beta_ext', sample_ext='pkl',
                       z_offset=0, vol_path=None):

        if vol_path is not None:
            self.vol_path = vol_path

        if data is not None:
            data = np.transpose(data, (1, 2, 0))
        elif sample_path is None:
            data = np.zeros((1, 1, 1, 1))
        else:
            if sample_ext is None:
                sample_path = sample_path
            else:
                sample_path = sample_path # + '.' + sample_ext
            with open(sample_path, "rb") as f:
                sample = pickle.load(f)
            data = np.transpose(sample[param_type], (2, 1, 0)) # smaple is [Z,Y,X]
            # data = np.transpose(sample[param_type], (1, 2, 0)) # smaple is [Z,Y,X]


        # Ensure that the data is a 4D numpy array with shape (Y, X, Z, channels)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=3)
        elif len(data.shape) != 4:
            raise ValueError("Data should be a 4D numpy array with shape (X, Y, Z, channels)")
        # Ensure that the data type is float32
        if data.dtype != np.float32:
            data = np.fliplr(data.astype(np.float32))

        dir = os.path.dirname(self.vol_path)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(self.vol_path, "wb") as f:
            # Write the header information
            f.write(b"VOL\x03")
            f.write(struct.pack("<i", 1))  # Encoding identifier
            f.write(struct.pack("<i", data.shape[2]))  # Number of cells along X
            f.write(struct.pack("<i", data.shape[0]))  # Number of cells along Y
            f.write(struct.pack("<i", data.shape[1]))  # Number of cells along Z
            f.write(struct.pack("<i", data.shape[3]))  # Number of channels
            # --- FIX 2: WRITE DIMENSIONS CORRECTLY (X, Y, Z) ---
            # # data is now (X, Y, Z, C)
            # f.write(struct.pack("<i", data.shape[1]))  # Size y
            # f.write(struct.pack("<i", data.shape[0]))  # Size Y
            # f.write(struct.pack("<i", data.shape[2]))  # Size Z
            # f.write(struct.pack("<i", data.shape[3]))  # Channels

            # Compute the bounding box of the data
            bbox = np.array([0, 0, 0, data.shape[2], data.shape[0], data.shape[1]], dtype=np.float32)
            #bbox = np.array([0, 0, 0, data.shape[2], data.shape[1], data.shape[0]], dtype=np.float32)

            f.write(struct.pack("<6f", *bbox))

            # Write the binary data
            data.tofile(f)

        if not self.centralize_cloud:
            if not (sample_path is None):
                z = sample['z'][0]
                self.cloud_zcenter = ((z[0] + z[-1]) / 2 + z_offset) / 1000
            self.update_required = True

    def set_the_scene(self, timestamp=None):

        if timestamp is not None:
            s_azimuth = self.sun_azimuth[timestamp]
            s_zenith = self.sun_zenith[timestamp]
        else:
            s_azimuth = self.sun_azimuth
            s_zenith = self.sun_zenith

        # --- Corrected Sun Direction Math ---
        # Convert degrees to radians
        az_rad = np.deg2rad(s_azimuth)
        ze_rad = np.deg2rad(s_zenith)

        # Standard spherical to cartesian "direction to" vector
        dir_x = -np.sin(ze_rad) * np.sin(az_rad)
        dir_y = -np.sin(ze_rad) * np.cos(az_rad)
        dir_z = np.cos(ze_rad)

        scene_dict = {
            'type': 'scene',
            'integrator': {  # integrator for volumes. max_depth is -1 for maximal accuracy
                'type': 'volpath',  # 'prbvolpath','volpath'
                'max_depth': -1,
                'rr_depth': 1000},
            # --- [NEW BLOCK] ---
            # This object represents the low-density "air"
            # that fills the entire scene.
            # 'atmosphere': {
            #     'type': 'cube',
            #     'bsdf': {'type': 'null'},  # Invisible container
            #     # This transform creates a massive 1000km-wide cube
            #     # centered at the world origin, filling the whole scene.
            #     'to_world': mi.scalar_rgb.Transform4f.scale(1000),
            #     'interior': {
            #         'type': 'homogeneous',
            #         'albedo': {
            #             'type': 'rgb',
            #             'value': 1.0
            #         },  # Pure white scattering
            #         'phase': {
            #             'type': 'hg',
            #             'g': 0.0  # Isotropic scattering (g=0) is a good
            #             # approximation for air, scattering light
            #             # equally in all directions.
            #         },
            #         'sigma_t': {
            #             'type': 'rgb',
            #             'value': 0.00001
            #         }
            #     }
            # },
            # --- [END OF MODIFICATION] ---
            'object': {  # transparent cube to contain our volume. The interior is the VOL we wrote
                'type': 'cube',
                'bsdf': {'type': 'null'},
                'to_world': mi.scalar_rgb.Transform4f.scale(self.W / 2 * 1e3 / self.scene_scale).translate(
                     [0, 0, 2 * self.cloud_zcenter]),
                #'to_world': mi.scalar_rgb.Transform4f.scale(self.W / 2 * 1e3 / self.scene_scale).rotate([1, 0, 0], 0),
                'interior': {
                    'type': 'heterogeneous',
                    'albedo': 1.0,
                    'phase': {
                        'type': 'hg',
                        'g': self.g_value
                    },
                    'sigma_t': {
                        'type': 'gridvolume',
                        'filename': self.vol_path,
                        'to_world': mi.scalar_rgb.Transform4f.rotate([0, 1, 0], -90).scale(
                        #'to_world': mi.scalar_rgb.Transform4f.scale(
                            self.W * 1e3 / self.scene_scale).translate(
                            [-0.5 + self.cloud_zcenter, -0.5, -0.5]),
                    },
                    'scale': self.scene_scale
                }
            },
            # 'emitter': {'type': 'constant'}, # constant lighting for testing
            'emitter': {  # Distant directional emitter - emulated the sun
                'type': 'directional',
                'direction': [dir_x, dir_y, dir_z],  # <-- Use corrected direction
                'irradiance': {
                    'type': 'rgb',
                    'value': 131.4,
                }
            },

            'ocean': {  # trying to emulate the ocean
                'type': 'cube',
                'to_world': mi.scalar_rgb.Transform4f.scale((self.W / 2 + 4) * 1e3 / self.scene_scale).translate(
                    [0, 0, -(self.W / 2 + 4) * 1e3 / self.scene_scale]),
                'bsdf': {  # Smooth diffuse BSDF
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'rgb',
                        'value': 0.0003
                    }
                }
            }
        }
        return scene_dict

    def set_scenes(self):

        if self.dynamic_emitter:
            for timestamp in range(self.timestamps):
                ref_scene = self.set_the_scene(timestamp=timestamp)
                self.scenes_dict.append(ref_scene)
                self.scenes.append(mi.load_dict(ref_scene))
        else:
            ref_scene = self.set_the_scene(timestamp=None)
            self.scenes_dict.append(ref_scene)
            self.scenes.append(mi.load_dict(ref_scene))

    def update_scenes(self, g_value=None):
        if g_value is not None:
            self.g_value = g_value
            self.update_required = True
        if self.update_required:
            if self.dynamic_emitter:
                for timestamp in range(self.timestamps):
                    ref_scene = self.set_the_scene(timestamp=timestamp)
                    self.scenes_dict[timestamp] = ref_scene
                    self.scenes[timestamp] = mi.load_dict(ref_scene)
            else:
                ref_scene = self.set_the_scene(timestamp=None)
                self.scenes_dict[0] = ref_scene
                self.scenes[0] = mi.load_dict(ref_scene)
        else:
            if self.dynamic_emitter:
                for timestamp in range(self.timestamps):
                    self.scenes[timestamp] = mi.load_dict(self.scenes_dict[timestamp])
            else:
                self.scenes[0] = mi.load_dict(self.scenes_dict[0])

    def render_scenes(self, spp=None):
        if spp is not None:
            self.spp = spp

        tensor_stacks = []
        bitmap_stacks = []

        for time_part in range(self.timestamps):
            tensors, bitmaps = self.render_scene(time_part, spp=None)
            tensor_stacks.append(tensors)
            bitmap_stacks.append(bitmaps)
        return tensor_stacks, bitmap_stacks

    def render_scene(self, time_part, spp=None):
        if spp is not None:
            self.spp = spp

        tensors = np.array([]).reshape((0, self.film_dim, self.film_dim))
        bitmaps = np.array([]).reshape((0, self.film_dim, self.film_dim))
        for sat in range(self.satellites):
            im_raw = mi.render(self.scenes[time_part * self.dynamic_emitter],
                               sensor=self.sensors[time_part * self.satellites + sat], spp=self.spp)
            im_gray = im_raw[:, :, 0]
            # im_gray = np.array(
            #     Image.fromarray((np.array(im_raw) / np.max(im_raw) * 255).astype(np.uint8)).convert('L'))
            ##### should I divide by 255 again?
            tensors = np.concatenate((tensors, np.expand_dims(np.array(im_gray), 0)), axis=0)

            if self.bitmaps_required:
                bitmap_raw = mi.util.convert_to_bitmap(im_raw)  # , 'uint8_srgb')
                bitmap_gray = np.array(Image.fromarray(np.array(bitmap_raw)).convert('L'))
                bitmaps = np.concatenate((bitmaps, np.expand_dims(bitmap_gray, 0)), axis=0)
            else:
                bitmaps = None

        return tensors, bitmaps
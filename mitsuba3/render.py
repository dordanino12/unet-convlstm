import mitsuba as mi
import numpy as np
import pandas as pd
import os


def setup_mitsuba_variant(variant='scalar_rgb'):
    """
    Set the Mitsuba 3 variant.
    'scalar_rgb' -> CPU-based, good for testing.
    'cuda_ad_rgb' -> GPU-based, much faster if you have an NVIDIA GPU.
    """
    try:
        mi.set_variant(variant)
        print(f"Mitsuba variant set to: {variant}")
    except Exception as e:
        print(f"Error setting Mitsuba variant: {e}")
        print("Please ensure Mitsuba 3 is installed correctly.")
        exit()


def create_dummy_csv(filepath='viewing_geo.csv'):
    """
    Creates a dummy CSV file with one row of viewing geometry.
    All coordinates are in ENU (East-North-Up).
    """
    print(f"Creating dummy CSV file at: {filepath}")
    data = {
        'utc time': ['2025-01-01T12:00:00'],
        'sun zenith [deg]': [30.0],
        'sun azimuth [deg]': [180.0],  # 180 deg = from the South
        'sat zenith [deg]': [10.0],  # Nearly overhead
        'sat azimuth [deg]': [0.0],
        'scattering angle [deg]': [140.0],
        'sat ENU coordinates [km]': [[2, 2, 2]],  # 800 km altitude
        'lookat ENU coordinates [km]': [[0, 0, 0]]  # Looking at origin
    }

    # Workaround for pandas storing lists in cells
    df = pd.DataFrame(data)
    df['sat ENU coordinates [km]'] = df['sat ENU coordinates [km]'].astype(str)
    df['lookat ENU coordinates [km]'] = df['lookat ENU coordinates [km]'].astype(str)

    df.to_csv(filepath, index=False)


def load_viewing_geometry(filepath, row_index=0):
    """
    Loads a specific row from the viewing geometry CSV file.
    Converts units (km -> m, deg -> rad).
    """
    print(f"Loading viewing geometry from: {filepath}")
    if not os.path.exists(filepath):
        create_dummy_csv(filepath)

    df = pd.read_csv(filepath)
    if row_index >= len(df):
        print(f"Error: Row index {row_index} is out of bounds for CSV file.")
        return None

    row = df.iloc[row_index]

    # Helper to parse the coordinate strings
    def parse_coord(coord_str):
        return np.array([float(c) for c in coord_str.strip('[]').split(',')])

    geo = {
        'sun_zenith_rad': np.radians(row['sun zenith [deg]']),
        'sun_azimuth_rad': np.radians(row['sun azimuth [deg]']),
        'sat_pos_m': parse_coord(row['sat ENU coordinates [km]']) * 1000.0,
        'lookat_pos_m': parse_coord(row['lookat ENU coordinates [km]']) * 1000.0,
    }
    return geo


def get_sun_direction(zenith_rad, azimuth_rad):
    """
    Converts spherical coordinates (zenith, azimuth) to a Cartesian direction vector.
    Assumes ENU: Azimuth from East (X), Zenith from Up (Z).
    The vector points *from* the sun *towards* the origin.
    """
    x = np.sin(zenith_rad) * np.cos(azimuth_rad)
    y = np.sin(zenith_rad) * np.sin(azimuth_rad)
    z = np.cos(zenith_rad)

    # The emitter's direction is where the light is *going*
    # So we reverse the vector (from origin towards sun)
    direction_to_sun = -np.array([x, y, z])
    return mi.Vector3f(direction_to_sun)


def generate_dummy_cloud_field(shape=(512, 512, 200)):
    """
    *** PLACEHOLDER FUNCTION ***
    This function generates a dummy 3D cloud field.
    You MUST replace this with your own function to load your data
    (e.g., from .npy, .vdb, or other formats).

    The output should be a 3D NumPy array of shape (512, 512, 200)
    representing the scattering density.
    """
    print(f"Generating dummy 3D cloud field of shape {shape}")
    print("!!! YOU MUST REPLACE THIS with your actual data loader !!!")

    # Create a simple "slab" of cloud
    x, y, z = shape
    density = np.zeros(shape, dtype=np.float32)

    # Define a cloud layer between z=50 and z=150
    cloud_bottom_idx = 50
    cloud_top_idx = 150

    density[:, :, cloud_bottom_idx:cloud_top_idx] = 0.5  # Base density

    # Add some simple noise for variation
    # Create coordinates
    X, Y, Z = np.meshgrid(
        np.linspace(-1, 1, y),
        np.linspace(-1, 1, x),
        np.linspace(-1, 1, z),
        indexing='xy'
    )

    # A simple procedural noise
    noise = (np.sin(X * 10) * np.cos(Y * 10) + np.sin(Z * 5)) * 0.5 + 0.5
    noise_slab = noise[:, :, cloud_bottom_idx:cloud_top_idx] * 0.5

    density[:, :, cloud_bottom_idx:cloud_top_idx] += noise_slab

    # Clip to [0, max_density]
    density = np.clip(density, 0, 1.0)

    # Scale density to be physically plausible
    # This 'max_density_scale' is a critical parameter to tune.
    # It represents the max extinction coefficient (per meter).
    max_density_scale = 0.1  # 0.1/m

    return density * max_density_scale


def main():
    # --- 1. Setup ---
    setup_mitsuba_variant('scalar_rgb')

    # --- 2. Define Scene Parameters ---
    csv_file = 'viewing_geo.csv'
    scene_index = 0  # Use the first row from the CSV

    # Your grid parameters
    grid_shape_voxels = (512, 512, 200)
    voxel_size_m = (20.0, 20.0, 20.0)

    # Total world size of the grid in meters
    world_size_m = np.array(grid_shape_voxels) * np.array(voxel_size_m)
    world_extents_m = world_size_m / 2.0  # Half-extents from center

    # --- 3. Load Data ---
    geo = load_viewing_geometry(csv_file, scene_index)
    if geo is None:
        return

    sun_direction = get_sun_direction(geo['sun_zenith_rad'], geo['sun_azimuth_rad'])

    # Load your 3D data
    # *** REPLACE THIS CALL ***
    # This should return a (512, 512, 200) np.float32 array.
    # The values should be the scattering density (sigma_s) in units of 1/meter.
    # You might need to scale your raw 'QV' or 'QR' data.
    density_data_numpy = generate_dummy_cloud_field(grid_shape_voxels)

    # Convert NumPy array to a Mitsuba VolumeGrid object
    # Mitsuba's VolumeGrid constructor expects data in (z, y, x) order.
    # Let's ensure our data is (x, y, z) and then transpose if needed.
    # The dummy data is (x, y, z), so we transpose to (z, y, x).
    density_volume = mi.VolumeGrid(density_data_numpy.transpose(2, 1, 0))

    # --- 4. Define Mitsuba Scene Dictionary ---
    print("Defining Mitsuba scene...")

    # Center the cloud box at z = half_height, so it rests on the z=0 plane
    box_center_z = world_extents_m[2]

    print(mi.Point3f(geo['sat_pos_m']))
    print(mi.Point3f(geo['lookat_pos_m']))
    print(mi.Point3f([0, 1, 0]))

    scene_dict = {
        'type': 'scene',

        # Integrator: Volumetric Path Tracer
        'integrator': {
            'type': 'volpath',
            'max_depth': 16,
        },

        # Emitter: The Sun
        'emitter': {
            'type': 'directional',
            'direction': sun_direction,
            # Adjust irradiance to make the scene brighter/dimmer
            'irradiance': {'type': 'rgb', 'value': 20.0},
        },

        # Sensor: The Satellite
        'sensor': {
            'type': 'perspective',
            'fov': 30,  # Field of View (degrees), adjust as needed
            'to_world': mi.Transform4f.look_at(
                mi.Point3f(geo['sat_pos_m']),
                mi.Point3f(geo['lookat_pos_m']),
                mi.Point3f([0, -1, 0])  # This is the fix, matching the definition
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 768,
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 64,  # Increase for less noise, but longer renders
            },
        },

        # Medium: The Cloud
        # We define this as a named medium to be referenced by the box
        'cloud_medium': {
            'type': 'heterogeneous',
            'scale': 1.0,  # You can scale the density array here
            'sigma_a': {'type': 'rgb', 'value': 0.0},  # No absorption (white cloud)

            # Scattering (sigma_s) is defined by our 3D grid
            'sigma_s': {
                'type': 'grid',
                'grid': '$density_volume_param',  # References the object we pass in
                # This transform maps the [0, 1]^3 grid coordinates
                # to the full world-space box.
                'to_world': mi.Transform4f.translate(mi.Vector3f([
                    -world_extents_m[0],
                    -world_extents_m[1],
                    0
                ]))
                .scale(mi.Vector3f([
                    world_size_m[0],
                    world_size_m[1],
                    world_size_m[2]
                ])),
            },

            # Phase Function: How light scatters inside the medium
            'phase': {
                'type': 'henyey_greenstein',
                'g': 0.85,  # Strong forward scattering, typical for clouds
            },
        },

        # Shape: The Box that contains the cloud medium
        'cloud_box': {
            'type': 'cube',

            # Assign the medium to the *inside* of the box
            'interior': {
                'type': 'ref',
                'id': 'cloud_medium',  # References the medium named 'cloud_medium'
            },

            # This transform defines the box's position and size in the world.
            # We scale a -1..+1 cube to our full world extents.
            'to_world': mi.Transform4f.translate([0, 0, box_center_z])
            .scale(world_extents_m),
        },
    }

    # --- 5. Load Scene and Render ---
    print("Loading scene into Mitsuba...")
    # Pass the Python VolumeGrid object to `load_dict`
    # The key 'density_volume_param' must match the '$' variable in the dict
    try:
        scene = mi.load_dict(
            scene_dict,
            density_volume_param=density_volume
        )
    except Exception as e:
        print(f"Error loading scene dictionary: {e}")
        print("This often happens if the 'density_volume_param' does not match")
        print("or if the Mitsuba variant is not set correctly.")
        return

    print("Starting render (this may take a while)...")
    # The 'spp' (samples per pixel) here overrides the sampler's 'sample_count'
    image = mi.render(scene, spp=64)

    # --- 6. Save Output ---
    output_filename = 'cloud_render.png'
    print(f"Render complete. Saving image to: {output_filename}")
    mi.util.write_bitmap(output_filename, image)
    print("Done.")


if __name__ == "__main__":
    main()






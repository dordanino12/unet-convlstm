from netCDF4 import Dataset
import pickle

# Open a NetCDF file
nc_file = Dataset('/wdata_visl/udigal/netCDF_20X20/BOMEX_512x512x200_20m_20m_1s_512_0000002040.nc', 'r')

# List dimensions
print(nc_file.dimensions.keys())  # e.g., 'time', 'lat', 'lon'

# List variables
print(nc_file.variables.keys())   # e.g., 'temperature', 'pressure'

# 'z' is both a dimension name and a variable name
# We access the 'z' variable via the 'variables' dictionary
z_variable = nc_file.variables['z']

# Using [:] loads all values from the variable into an array (usually a numpy array)
z_values = z_variable[:]

print("--- Values of the 'z' variable ---")
print(z_values)
print("----------------------------------")

x_variable = nc_file.variables['x']

# Using [:] loads all values from the variable into an array (usually a numpy array)
x_values = x_variable[:]

print("--- Values of the 'x' variable ---")
print(x_values)
print("----------------------------------")

nc_file.close()

pkl_file = "/wdata_visl/udigal/samples/samples_mode3_res128_stride64_spp8/samples_3D/BOMEX_512x512x200_20m_20m_1s_512_0000002000_0_3.pkl"
with open(pkl_file, "rb") as f:
    data = pickle.load(f)
    print(data)


from netCDF4 import Dataset

# Open a NetCDF file
nc_file = Dataset('/wdata_visl/udigal/netCDF_20X20/BOMEX_512x512x200_20m_20m_1s_512_0000002040.nc', 'r')

# List dimensions
print(nc_file.dimensions.keys())  # e.g., 'time', 'lat', 'lon'

# List variables
print(nc_file.variables.keys())   # e.g., 'temperature', 'pressure'

nc_file.close()

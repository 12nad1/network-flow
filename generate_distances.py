import numpy as np
import xarray as xr
from geopy.distance import geodesic
from itertools import product

ds = xr.load_netcdf('L.T.iron_flows.2000-2016.a.nc')

# Select time slice
ds_time = ds.sel(time=time)

# Extract coordinates
lats = ds_time.lat.values
lons = ds_time.lon.values

# Create all coordinate pairs (lat1, lon1) x (lat2, lon2)
coords1 = list(product(lats, lons))
coords2 = coords1  # since you're comparing all pairs

# Prepare arrays to hold output
lat1_list, lon1_list, lat2_list, lon2_list, dist_list = [], [], [], [], []

for (lat1, lon1), (lat2, lon2) in product(coords1, coords2):
    lat1_list.append(lat1)
    lon1_list.append(lon1)
    lat2_list.append(lat2)
    lon2_list.append(lon2)
    
    if lat1 == lat2 and lon1 == lon2:
        dist_list.append(0.0)
    else:
        dist_list.append(geodesic((lat1, lon1), (lat2, lon2)).km)

# Convert to xarray Dataset
out_ds = xr.Dataset({
    "lat1": (["pairs"], lat1_list),
    "lon1": (["pairs"], lon1_list),
    "lat2": (["pairs"], lat2_list),
    "lon2": (["pairs"], lon2_list),
    "distance": (["pairs"], dist_list)
})

out_ds.to_netcdf('distances.nc')

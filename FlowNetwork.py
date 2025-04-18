import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import sesame as ssm

class FlowNetwork:
	def __init__(self, netcdf_path, outflow_var, inflow_var, node_thresh=0, time=None):
		self.ds = xr.load_dataset(netcdf_path)
		if time:
			self.ds = self.ds.sel(time=time).squeeze(drop=True)
		self.inflow_var = inflow_var
		self.outflow_var = outflow_var
		self.node_thresh = node_thresh
		self.source_file = netcdf_path
		self.selected_time = time

		# Extract valid coordinates
		df = (
			self.ds
			.reset_coords(drop=True)
			.stack(points=("lat", "lon"))
			.to_dataframe()
		)
		self.df = df.dropna(subset=[self.inflow_var, self.outflow_var], how='all')

		self.in_coords = df.index[(df[self.inflow_var].notna()) & (df[self.inflow_var] > node_thresh)].tolist()
		self.out_coords = df.index[(df[self.outflow_var].notna()) & (df[self.outflow_var] > node_thresh)].tolist()

		self.coord_to_in_idx = {coord: i for i, coord in enumerate(self.in_coords)}
		self.coord_to_out_idx = {coord: i for i, coord in enumerate(self.out_coords)}
		self.in_idx_to_coord = {i: coord for coord, i in self.coord_to_in_idx.items()}
		self.out_idx_to_coord = {i: coord for coord, i in self.coord_to_out_idx.items()}

		self.ensemble = np.zeros((len(self.out_coords), len(self.in_coords)))
		self.flow = np.zeros_like(self.ensemble)


	## Helper Functions ##

	def coord_to_index(self, coord, kind='in'):
		if kind == 'in':
			return self.coord_to_in_idx[coord]
		elif kind == 'out':
			return self.coord_to_out_idx[coord]

	def index_to_coord(self, idx, kind='in'):
		if kind == 'in':
			return self.in_idx_to_coord[idx]
		elif kind == 'out':
			return self.out_idx_to_coord[idx]
		

	## Algorithm Methods ##
		
	def pairwise_haversine(self, in_coords, out_coords):
		"""
		Compute the pairwise haversine distances between all in_coords and out_coords.

		Parameters:
			in_coords (np.ndarray): Shape (M, 2) for M inflow coordinates
			out_coords (np.ndarray): Shape (N, 2) for N outflow coordinates

		Returns:
			distances (np.ndarray): Shape (M, N) where dist[i, j] is distance from in_coords[i] to out_coords[j]
		"""
		lat1 = np.radians(in_coords[:, 0])[:, np.newaxis]  # shape (I, 1)
		lon1 = np.radians(in_coords[:, 1])[:, np.newaxis]
		lat2 = np.radians(out_coords[:, 0])[np.newaxis, :]  # shape (1, O)
		lon2 = np.radians(out_coords[:, 1])[np.newaxis, :]

		dlat = np.abs(lat2 - lat1)  # shape (I, O)
		dlon = np.abs(lon2 - lon1)
		dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)  # Correct for wraparound in longitude
    
		a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
		c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) #TODO: Fix invalid value sometimes

		return 6371 * c  # Earth's radius in km


	def optimized_gravity_model(self):
		print("Running Gravity Model...")
		self.ensemble[:] = 0  # Reset

		inflow_values = self.df.loc[self.in_coords][self.inflow_var].values  # shape (I,)
		outflow_values = self.df.loc[self.out_coords][self.outflow_var].values  # shape (O,)

		in_coords_array = np.array(self.in_coords)  # shape (I, 2)
		out_coords_array = np.array(self.out_coords)  # shape (O, 2)

		distances = self.pairwise_haversine(out_coords_array, in_coords_array)  # shape (O, I)
		distances[distances == 0] = np.nan  # prevent divide-by-zero

		gravity_matrix = np.outer(outflow_values, inflow_values) / distances  # shape (O, I)
		gravity_matrix = np.nan_to_num(gravity_matrix)  # replace nan with 0

		self.ensemble = gravity_matrix

		print("Gravity Model Complete. Ensemble Generated.")


	def apply_ipf_to_ensemble(self, max_iters=100, tol=1e-6):
		print("Running IPF...")

		# Make sure target arrays match ensemble matrix shape
		outflow_targets = self.df.loc[self.out_coords, self.outflow_var].values  # (O,)
		inflow_targets = self.df.loc[self.in_coords, self.inflow_var].values     # (I,)

		outflow_targets += 1e-10  # prevent divide by zero
		inflow_targets += 1e-10

		W = self.ensemble.copy()  # Shape: (O, I)

		for _ in range(max_iters):
			# Scale rows (outflows)
			row_sums = W.sum(axis=1)  # (O,)
			W *= (outflow_targets / row_sums)[:, None]  # Broadcast: (O, 1)

			# Scale columns (inflows)
			col_sums = W.sum(axis=0)  # (I,)
			W *= (inflow_targets / col_sums)[None, :]  # Broadcast: (1, I)

			# Check for convergence
			if np.allclose(W.sum(axis=1), outflow_targets, atol=tol) and \
			np.allclose(W.sum(axis=0), inflow_targets, atol=tol):
				break

		self.flow = W
		print("IPF complete. Flow matrix generated.")


	def get_flow_value(self, in_coord, out_coord):
		i = self.coord_to_in_idx[in_coord]
		j = self.coord_to_out_idx[out_coord]
		return self.flow[i, j]

	def get_ensemble_value(self, in_coord, out_coord):
		i = self.coord_to_in_idx[in_coord]
		j = self.coord_to_out_idx[out_coord]
		return self.ensemble[i, j]
	
	def plot_network_map(self, use='flow', num_edges=1000, vmin=None, vmax=None, levels=6, edge_thickness=1, edge_alpha=0.5, edge_color='gray'):
		"""
		Plot the flow or ensemble network over a variable background map.

		Parameters:
		- use: 'flow' or 'ensemble'
		- vmin, vmax: limits for background variable
		- edge_scale: scale factor for edge line width
		- edge_alpha: transparency for edges
		"""
		if use not in ['flow', 'ensemble']:
			raise ValueError("use must be 'flow' or 'ensemble'")

		edges = self.flow if use == 'flow' else self.ensemble
		in_coords = self.in_coords
		out_coords = self.out_coords

		self.ds['netflow'] = self.ds[self.outflow_var] - self.ds[self.inflow_var]
		ax = ssm.plot_map('netflow', self.ds, levels=levels, vmin=vmin, vmax=vmax, color='coolwarm', show=False, remove_ata=True)

		# Get flat indices of top edges by weight
		flat_indices = np.argpartition(edges.ravel(), -num_edges)[-num_edges:]
		top_indices = flat_indices[np.argsort(edges.ravel()[flat_indices])][::-1]  # Sorted in descending order

		# Convert flat indices to 2D indices
		out_idxs, in_idxs = np.unravel_index(top_indices, edges.shape)

		edge_scale = edge_thickness / edges.max()

		# Plot only the strongest edges
		for out_i, in_j in zip(out_idxs, in_idxs):
			weight = edges[out_i, in_j]
			if weight > 0:
				out_coord = out_coords[out_i]
				in_coord = in_coords[in_j]
				start = (out_coord[1], out_coord[0])  # (lon, lat)
				end = (in_coord[1], in_coord[0])      # (lon, lat)
				ax.plot(
					[start[0], end[0]],
					[start[1], end[1]],
					color=edge_color,
					linewidth=weight * edge_scale,
					alpha=edge_alpha,
					transform=ccrs.Geodetic()
				)

		plt.title(f"{use.title()} Network {self.ds[self.inflow_var].attrs['units']}")
		plt.show()

	def compute_country_trade(self):
		"""
		Add estimated imports and exports to self.ds by comparing edge flows between different countries.
		Requires static_ctry_frac_ds with shape (lat, lon), one data_var per country.
		"""
		print("Computing country trade from network flows...")
		import os
		ctry_frac_path = os.path.join(ssm.__path__[0],'data','country_fraction.1deg.2000-2023.a.nc')
		ctry_frac_ds = xr.load_dataset(ctry_frac_path)
		static_ctry_frac_ds = ctry_frac_ds.sel(time='2000').squeeze(drop=True)

		# Initialize import/export arrays
		shape = self.ds[self.inflow_var].shape
		imports = np.zeros(shape)
		exports = np.zeros(shape)

		# Precompute dominant country per grid cell (for performance)
		ctry_stack = static_ctry_frac_ds.to_array(dim="country")  # shape: (country, lat, lon)
		dominant_country_idx = ctry_stack.argmax(dim="country")
		dominant_country = ctry_stack.country[dominant_country_idx]

		# Convert to dict of (lat, lon) -> country string
		lat_vals = static_ctry_frac_ds.lat.values
		lon_vals = static_ctry_frac_ds.lon.values
		country_map = {
			(float(lat), float(lon)): str(dominant_country.sel(lat=lat, lon=lon).values)
			for lat in lat_vals for lon in lon_vals
		}

		# Iterate through edges
		total_out = len(self.out_coords)
		for i, out_coord in enumerate(self.out_coords):
			print(f"progress: {i/total_out*100:.2f}%")
			for j, in_coord in enumerate(self.in_coords):
				flow_val = self.flow[i, j]
				if flow_val <= 0:
					continue

				# Ensure we only consider different countries
				out_country = country_map.get(tuple(map(float, out_coord)), None)
				in_country = country_map.get(tuple(map(float, in_coord)), None)

				if out_country and in_country and out_country != in_country:
					# Find indices of coords in ds
					lat_in, lon_in = in_coord
					lat_out, lon_out = out_coord

					# Find nearest index in dataset
					lat_in_idx = np.argmin(np.abs(self.ds.lat.values - lat_in))
					lon_in_idx = np.argmin(np.abs(self.ds.lon.values - lon_in))
					lat_out_idx = np.argmin(np.abs(self.ds.lat.values - lat_out))
					lon_out_idx = np.argmin(np.abs(self.ds.lon.values - lon_out))

					imports[lat_in_idx, lon_in_idx] += flow_val
					exports[lat_out_idx, lon_out_idx] += flow_val

		# Add the results to the dataset
		self.ds["estimated_imports"] = (("lat", "lon"), imports)
		self.ds["estimated_exports"] = (("lat", "lon"), exports)
		self.ds.to_netcdf('output.nc')

		df1 = ssm.grid_2_table(dataset=self.ds, variables="estimated_imports", agg_function='sum')
		df2 = ssm.grid_2_table(dataset=self.ds, variables="estimated_imports", agg_function='sum')
		merged_df = merge(df1,df2,on='ISO3')
		merged_df.to_csv('predicted_import_export.csv')
		print("✓ Estimated imports and exports and saved.")


	def __str__(self):
		lines = []
		lines.append("🌍 Grid Dataset Summary")
		lines.append(f"  NetCDF Source: {getattr(self, 'source_file', 'unknown')}")
		lines.append(f"  Time Selected: {getattr(self, 'selected_time', 'N/A')}")
		lines.append(f"  Inflow Var: {self.inflow_var} ({self.ds[self.inflow_var].attrs['units']})")
		lines.append(f"  Outflow Var: {self.outflow_var} ({self.ds[self.inflow_var].attrs['units']})")
		lines.append(f"  Total Flow (IO): {self.ds[self.outflow_var].sum().item():,.2f} {self.ds[self.outflow_var].sum().item():,.2f}")

		if hasattr(self, "ensemble") and isinstance(self.ensemble, np.ndarray):
			lines.append("\n📊 Ensemble Graph")
			lines.append(f"  Shape: {self.ensemble.shape}")
			lines.append(f"  Total Weight: {self.ensemble.sum():,.2f}")
			lines.append(f"  Nonzero Edges: {(self.ensemble > 0).sum()}")

		if hasattr(self, "flow") and isinstance(self.flow, np.ndarray):
			lines.append("\n📦 Flow Graph")
			lines.append(f"  Shape: {self.flow.shape}")
			lines.append(f"  Total Flow: {self.flow.sum():,.2f}")
			lines.append(f"  Nonzero Edges: {(self.flow > 0).sum()}")

		return "\n".join(lines)


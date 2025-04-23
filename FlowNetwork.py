import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import os
import sesame as ssm

class FlowNetwork:
	def __init__(self, dataset, outflow_var, inflow_var, node_thresh=0, time=None, verbose=False):
		if isinstance(dataset, str):
			self.ds = xr.load_dataset(dataset)
		elif isinstance(dataset, xr.Dataset):
			self.ds = dataset
		else:
			raise TypeError("Expected a string path or an xarray.Dataset, but got {}".format(type(dataset)))
		if time:
			self.ds = self.ds.sel(time=time).squeeze(drop=True)
		self.inflow_var = inflow_var
		self.outflow_var = outflow_var

		try:
			assert(self.ds[self.outflow_var].sum().values == self.ds[self.inflow_var].sum().values)
			self.total_flow = self.ds[self.outflow_var].sum().values
		except:
			if verbose:
				print(f"Warning: Total inflow ({self.ds[self.inflow_var].sum().values}) does not equal total outflow ({self.ds[self.outflow_var].sum().values})")
			self.total_flow = (self.ds[self.outflow_var].sum().values + self.ds[self.inflow_var].sum().values) / 2

		self.node_thresh = node_thresh
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

		# Ensemble contains a notion of affinity between nodes
		self.ensemble = np.zeros((len(self.out_coords), len(self.in_coords)))

		# Flow contains an instantiation of flow between nodes based on the ensemble affinities
		self.flow = np.zeros_like(self.ensemble)

		# Predicted marginal df contains the predicted imports/exports distribution of the flows in and out of each country
		self.predicted_marginal_df = None

	def save_data(self, output_path, verbose=False):
		"""
		Save the FlowNetwork data to a compressed file containing:
		- self.ds as a netCDF file
		- self.ensemble and self.flow as numpy arrays
		- self.predicted_marginal_df as a CSV file
		
		Parameters:
			output_path (str): Path to save the compressed file
			verbose (bool): Whether to print progress information
		"""
		import os
		import shutil
		import tempfile
		
		if verbose:
			print(f"Saving FlowNetwork data to {output_path}...")
			
		# Create temporary directory
		with tempfile.TemporaryDirectory() as temp_dir:
			# Save dataset as netCDF
			ds_path = os.path.join(temp_dir, 'dataset.nc')
			self.ds.to_netcdf(ds_path)
			
			# Save numpy arrays
			np.save(os.path.join(temp_dir, 'ensemble.npy'), self.ensemble)
			np.save(os.path.join(temp_dir, 'flow.npy'), self.flow)
			
			# Save predicted marginal dataframe if it exists
			if self.predicted_marginal_df is not None:
				self.predicted_marginal_df.to_csv(os.path.join(temp_dir, 'predicted_marginal.csv'))
			
			# Create metadata file
			metadata = {
				'inflow_var': self.inflow_var,
				'outflow_var': self.outflow_var,
				'node_thresh': self.node_thresh,
				'selected_time': str(self.selected_time) if self.selected_time else None
			}
			np.save(os.path.join(temp_dir, 'metadata.npy'), metadata)
			
			# Create zip file
			shutil.make_archive(output_path.replace('.zip', ''), 'zip', temp_dir)
			
		if verbose:
			print("✓ Data saved successfully")

	@classmethod
	def load_data(cls, input_path, verbose=False):
		"""
		Load a FlowNetwork from a compressed file.
		
		Parameters:
			input_path (str): Path to the compressed file
			verbose (bool): Whether to print progress information
			
		Returns:
			FlowNetwork: A new FlowNetwork instance with the loaded data
		"""
		import os
		import shutil
		import tempfile
		
		if verbose:
			print(f"Loading FlowNetwork data from {input_path}...")
			
		# Create temporary directory
		with tempfile.TemporaryDirectory() as temp_dir:
			# Extract zip file
			shutil.unpack_archive(input_path, temp_dir, 'zip')
			
			# Load dataset
			ds = xr.load_dataset(os.path.join(temp_dir, 'dataset.nc'))
			
			# Load metadata
			metadata = np.load(os.path.join(temp_dir, 'metadata.npy'), allow_pickle=True).item()
			
			# Create FlowNetwork instance
			flow_network = cls(
				dataset=ds,
				outflow_var=metadata['outflow_var'],
				inflow_var=metadata['inflow_var'],
				node_thresh=metadata['node_thresh'],
				time=metadata['selected_time'],
				verbose=verbose
			)
			
			# Load numpy arrays
			flow_network.ensemble = np.load(os.path.join(temp_dir, 'ensemble.npy'))
			flow_network.flow = np.load(os.path.join(temp_dir, 'flow.npy'))
			
			# Load predicted marginal dataframe if it exists
			predicted_marginal_path = os.path.join(temp_dir, 'predicted_marginal.csv')
			if os.path.exists(predicted_marginal_path):
				flow_network.predicted_marginal_df = pd.read_csv(predicted_marginal_path)
			
		if verbose:
			print("✓ Data loaded successfully")
			
		return flow_network

	## Helper Functions ##

	def _coord_to_index(self, coord, kind='in'):
		if kind == 'in':
			return self.coord_to_in_idx[coord]
		elif kind == 'out':
			return self.coord_to_out_idx[coord]

	def _index_to_coord(self, idx, kind='in'):
		if kind == 'in':
			return self.in_idx_to_coord[idx]
		elif kind == 'out':
			return self.out_idx_to_coord[idx]
				
	def _pairwise_haversine(self, in_coords, out_coords):
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

	def _limplot(self, df, x_col, y_col, x_log=False, y_log=False, rm_outliers=False):
		# Drop rows with missing values in x, y, or ISO3
		df_copy = df.dropna(subset=[x_col, y_col, "ISO3"]).copy()

		# Prepare x axis values: if x_log is True, compute log10 and filter positive values only.
		if x_log:
			# Filter out non-positive values (log10 only works for positive numbers)
			df_copy = df_copy[df_copy[x_col] > 0]
			df_copy["x_plot"] = np.log10(df_copy[x_col])
			x_label = f"log10({x_col})"
		else:
			df_copy["x_plot"] = df_copy[x_col]
			x_label = x_col

		# Prepare y axis values: if y_log is True, compute log10 and filter positive values only.
		if y_log:
			df_copy = df_copy[df_copy[y_col] > 0]
			df_copy["y_plot"] = np.log10(df_copy[y_col])
			y_label = f"log10({y_col})"
		else:
			df_copy["y_plot"] = df_copy[y_col]
			y_label = y_col

		# Optionally remove outliers using the IQR method on the columns to be plotted (x_plot and y_plot)
		if rm_outliers:
			# For x_plot values
			Q1_x = df_copy["x_plot"].quantile(0.25)
			Q3_x = df_copy["x_plot"].quantile(0.75)
			IQR_x = Q3_x - Q1_x
			lower_bound_x = Q1_x - 1.5 * IQR_x
			upper_bound_x = Q3_x + 1.5 * IQR_x

			# For y_plot values
			Q1_y = df_copy["y_plot"].quantile(0.25)
			Q3_y = df_copy["y_plot"].quantile(0.75)
			IQR_y = Q3_y - Q1_y
			lower_bound_y = Q1_y - 1.5 * IQR_y
			upper_bound_y = Q3_y + 1.5 * IQR_y

			# Remove rows where x or y are outside their respective bounds
			df_copy = df_copy[
				(df_copy["x_plot"] >= lower_bound_x) & (df_copy["x_plot"] <= upper_bound_x) &
				(df_copy["y_plot"] >= lower_bound_y) & (df_copy["y_plot"] <= upper_bound_y)
			]
		
		# Compute the R² value using a linear regression (fit on the x_plot and y_plot values)
		x_vals = df_copy["x_plot"].values
		y_vals = df_copy["y_plot"].values
		if len(x_vals) > 1:  # Ensure there is more than one data point
			slope, intercept = np.polyfit(x_vals, y_vals, 1)
			y_pred = slope * x_vals + intercept
			r2 = 1 - np.sum((y_vals - y_pred)**2) / np.sum((y_vals - np.mean(y_vals))**2)
		else:
			r2 = np.nan

		# Create the lmplot using the prepared (and optionally transformed) data
		g = sns.lmplot(
			x="x_plot", 
			y="y_plot", 
			data=df_copy,
			scatter_kws={"alpha": 0.6, "s": 15},
			line_kws={"color": "red"},
			ci=95
		)

		# Set the plot title to reflect any log transformation
		plt.title(f"{x_label} vs {y_label}")
		
		# Annotate the R² value on the plot (using relative axis coordinates)
		ax = g.axes[0, 0]
		# Override the axis labels to be the original variable names
		ax.set_xlabel(x_col)
		ax.set_ylabel(y_col)
		ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}",
				transform=ax.transAxes, fontsize=10, verticalalignment='top')

		# Annotate each point with the ISO3 code
		for i, row in df_copy.iterrows():
			ax.text(
				row["x_plot"], 
				row["y_plot"], 
				row["ISO3"],
				fontsize=8,
				alpha=0.8
			)

		plt.show()

	## Algorithm Methods ##

	def gravity_model(self, threshold_percentile=100, verbose=False):
		if verbose:
			if threshold_percentile < 100:
				print(f"Running Gravity Model with thresholding at {threshold_percentile}% of possible edges...")
			else:
				print("Running Gravity Model...")
		self.ensemble[:] = 0  # Reset

		inflow_values = self.df.loc[self.in_coords][self.inflow_var].values  # shape (I,)
		outflow_values = self.df.loc[self.out_coords][self.outflow_var].values  # shape (O,)

		in_coords_array = np.array(self.in_coords)  # shape (I, 2)
		out_coords_array = np.array(self.out_coords)  # shape (O, 2)

		distances = self.pairwise_haversine(out_coords_array, in_coords_array)  # shape (O, I)
		distances[distances == 0] = np.nan  # prevent divide-by-zero

		# Calculate gravity matrix g = G * m_o * m_i / d
		gravity_matrix = np.outer(outflow_values, inflow_values) / distances  # shape (O, I)
		gravity_matrix = np.nan_to_num(gravity_matrix)  # replace nan with 0
		if verbose:
			print(f"Edges after gravity calculation: {np.sum(gravity_matrix > 0)}")

		# Apply threshold if specified
		if threshold_percentile < 100:
			# Step 1: For each node, keep its strongest connections
			thresholded_gravity_matrix = np.zeros_like(gravity_matrix)
			
			# For each outflow node, keep its strongest connections
			for i in range(len(self.out_coords)):
				row = gravity_matrix[i, :]
				if np.any(row > 0):
					threshold = np.percentile(row[row > 0], 100 - threshold_percentile)
					thresholded_gravity_matrix[i, :] = np.where(row >= threshold, row, 0)
			
			# For each inflow node, keep its strongest connections
			for j in range(len(self.in_coords)):
				col = gravity_matrix[:, j]
				if np.any(col > 0):
					threshold = np.percentile(col[col > 0], 100 - threshold_percentile)
					thresholded_gravity_matrix[:, j] = np.maximum(thresholded_gravity_matrix[:, j], 
						np.where(col >= threshold, col, 0))
			
			# Step 2: Edge Data Analysis
			total_flow_mass = np.sum(gravity_matrix)
			nonzero_flow_mass = np.sum(thresholded_gravity_matrix[thresholded_gravity_matrix > 0])
			if verbose:
				print(f"After thresholding there are {np.sum(thresholded_gravity_matrix > 0)} edges containing {nonzero_flow_mass/total_flow_mass:.1%} of total flow mass")

			# Step 3: Node Analysis
			nonzero_rows = np.any(thresholded_gravity_matrix > 0, axis=1)
			nonzero_cols = np.any(thresholded_gravity_matrix > 0, axis=0)
			num_nodes = np.sum(nonzero_rows) + np.sum(nonzero_cols)
			if verbose:
				print(f"Number of nodes after thresholding: {num_nodes}")

			# Calculate flow contained in remaining nodes
			total_inflow = np.sum(inflow_values)
			total_outflow = np.sum(outflow_values)
			inflow_contained = np.sum(inflow_values[nonzero_cols])
			outflow_contained = np.sum(outflow_values[nonzero_rows])
			if verbose:
				print(f"Percent of inflow contained in remaining nodes: {inflow_contained/total_inflow:.1%}")
				print(f"Percent of outflow contained in remaining nodes: {outflow_contained/total_outflow:.1%}")

			gravity_matrix = thresholded_gravity_matrix

		self.ensemble = gravity_matrix
		if verbose:
			print("Gravity Model Complete. Ensemble Generated.")

	def ipf_flows(self, max_iters=100, tol=1e-6, verbose=False):
		"""
		Apply Iterative Proportional Fitting to match row and column sums.
		This method preserves the relative weights of the non-zero entries while
		matching the target row and column sums.
		"""
		if verbose:
			print("Running IPF...")
			print(f"Initial edges in ensemble: {np.sum(self.ensemble > 0)}")
		
		# Get target row and column sums
		target_row_sums = self.df.loc[self.out_coords][self.outflow_var].values
		target_col_sums = self.df.loc[self.in_coords][self.inflow_var].values
		
		# Initialize working matrix
		W = self.ensemble.copy()
		
		# Ensure we have non-zero entries for IPF
		if np.sum(W > 0) == 0:
			raise ValueError("No non-zero entries in ensemble matrix. IPF cannot proceed.")
		
		for iteration in range(max_iters):
			# Row scaling
			row_sums = W.sum(axis=1)
			row_sums[row_sums == 0] = 1  # prevent divide by zero
			W = W * (target_row_sums[:, np.newaxis] / row_sums[:, np.newaxis])
			
			# Column scaling
			col_sums = W.sum(axis=0)
			col_sums[col_sums == 0] = 1  # prevent divide by zero
			W = W * (target_col_sums[np.newaxis, :] / col_sums[np.newaxis, :])
			
			# Check convergence
			row_error = np.max(np.abs(W.sum(axis=1) - target_row_sums))
			col_error = np.max(np.abs(W.sum(axis=0) - target_col_sums))
			
			if max(row_error, col_error) < tol:
				if verbose:
					print(f"IPF converged in {iteration + 1} iterations")
				break
		
		self.flow = W
		if verbose:
			print(f"IPF complete. Flow matrix generated with {np.sum(self.flow > 0)} edges.")
	
	def plot_network_map(self, caption=None, num_edges=None, vmin=None, vmax=None, color='coolwarm', levels=6, edge_thickness=1, edge_alpha=0.5, edge_color='gray'):
		"""
		Plot the flow network over a variable background map.

		Parameters:
		- num_edges: number of edges to plot, if None, all edges are plotted
		- vmin, vmax: limits for background variable
		- edge_scale: scale factor for edge line width
		- edge_alpha: transparency for edges
		"""

		edges = self.flow
		in_coords = self.in_coords
		out_coords = self.out_coords

		self.ds['netflow'] = self.ds[self.outflow_var] - self.ds[self.inflow_var]
		ax = ssm.plot_map('netflow', self.ds, levels=levels, vmin=vmin, vmax=vmax, color=color, show=False, remove_ata=True)

		# Count non-zero edges
		non_zero_edges = np.sum(edges > 0)
		if num_edges is None or num_edges > non_zero_edges:
			num_edges = non_zero_edges

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

		plt.title(f"{'flow'.title()} Network {self.ds[self.inflow_var].attrs['units']}")
		
		visualized_flow = edges.ravel()[top_indices].sum()
		percentage = (visualized_flow / self.total_flow) * 100
		added_text_if_edges_dropped = f"out of {non_zero_edges}" if num_edges is not None else ""
		plt.figtext(0.5, 0.01, f"Visualized {num_edges} {added_text_if_edges_dropped} edges containing {visualized_flow:.1f} {self.ds[self.inflow_var].attrs['units']} ({percentage:.1f}% of total flow) {caption}", 
			ha='center', fontsize=10)
		plt.show()

	def compute_country_trade(self, out_path=None, verbose=False):
		"""
		Add estimated imports and exports to self.ds by comparing edge flows between different countries.
		Requires static_ctry_frac_ds with shape (lat, lon), one data_var per country.
		"""
		if verbose:
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
			if verbose:
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

		df1 = ssm.grid_2_table(dataset=self.ds, variables="estimated_imports", agg_function='sum')
		df2 = ssm.grid_2_table(dataset=self.ds, variables="estimated_imports", agg_function='sum')
		self.predicted_marginal_df = pd.merge(df1,df2,on='ISO3')
		if out_path:
			self.predicted_marginal_df.to_csv(out_path)
			if verbose:
				print("✓ Estimated imports and exports and saved.")
	
	def check_mass_conservation(self, tol=1e-6, verbose=False):
		"""
		Verify that the flow matrix conserves mass at each node, by comparing
		inflow and outflow values in the dataset to the sums of incoming and
		outgoing edges in the flow matrix.
		"""

		if verbose:
			print("Checking mass conservation...")

		inflow_pass, outflow_pass = True, True

		# Check inflow nodes: sum over columns (axis=1)
		for j, in_coord in enumerate(self.in_coords):
			flow_in = self.flow[:, j].sum()
			ds_in = self.df.loc[in_coord][self.inflow_var]
			if not np.isclose(flow_in, ds_in, atol=tol):
				inflow_pass = False
				print(f"[!] Inflow mismatch at {in_coord}: graph={flow_in:.3f}, dataset={ds_in:.3f}")

		# Check outflow nodes: sum over rows (axis=0)
		for i, out_coord in enumerate(self.out_coords):
			flow_out = self.flow[i, :].sum()
			ds_out = self.df.loc[out_coord][self.outflow_var]
			if not np.isclose(flow_out, ds_out, atol=tol):
				outflow_pass = False
				print(f"[!] Outflow mismatch at {out_coord}: graph={flow_out:.3f}, dataset={ds_out:.3f}")

		if inflow_pass and outflow_pass:
			if verbose:
				print("✅ All nodes pass mass conservation check.")
		else:
			print("❌ Some nodes failed the check. See above for details.")

	def analyze_degree_distribution(self, caption=None, verbose=False):
		"""
		Analyze the degree distribution of the flow network by fitting various probability distributions.
		Returns a dictionary containing the fitted parameters and statistics.
		"""
		import numpy as np
		import scipy.stats as stats
		from collections import OrderedDict
		import matplotlib.pyplot as plt

		# Calculate degree sequence from flow matrix
		degree_sequence = np.sum(self.flow > 0, axis=1)  # Out-degree for each node
		degree_sequence = degree_sequence[degree_sequence > 0]  # Remove zero-degree nodes

		# Distributions to fit
		distributions = OrderedDict({
			'normal': stats.norm,
			'lognorm': stats.lognorm,
			'expon': stats.expon,
			'powerlaw': stats.powerlaw
		})

		fitted_results = {}

		# Fit the different distributions
		for name, dist in distributions.items():
			try:
				params = dist.fit(degree_sequence)
				log_likelihood = np.sum(dist.logpdf(degree_sequence, *params))
				num_params = len(params)
				aic = 2 * num_params - 2 * log_likelihood
				fitted_results[name] = {
					'params': params,
					'log_likelihood': log_likelihood,
					'aic': aic,
					'num_params': num_params
				}
			except Exception as e:
				print(f"Could not fit {name}: {e}")

		# Plot the results
		plt.figure(figsize=(10, 6))
		
		# Create 15 equal-sized bins between min and max degree
		min_degree = min(degree_sequence)
		max_degree = max(degree_sequence)
		if verbose:
			print('Min degree:', min_degree, 'Max degree:', max_degree)
		bins = np.linspace(min_degree, max_degree, 16)  # 16 edges for 15 bins
		
		# Plot histogram with equal-sized bins
		hist, bins, _ = plt.hist(degree_sequence, bins=bins, density=False,
								alpha=0.6, label='Data', color='gray')
		
		x = np.linspace(min(degree_sequence), max(degree_sequence), 100)
		for name, res in fitted_results.items():
			if 'params' in res:
				# Scale the PDF to match the histogram counts
				y = distributions[name].pdf(x, *res['params']) * len(degree_sequence) * (bins[1] - bins[0])
				plt.plot(x, y, label=f"{name} (AIC={res['aic']:.0f})")

		if caption:
			plt.figtext(0.5, 0.01, caption, ha='center', fontsize=10)

		plt.yscale('log')
		plt.xlabel('Degree')
		plt.ylabel('Number of Nodes')
		plt.title('Degree Distribution with Fitted Models')
		plt.legend()
		plt.grid(True, which="both", ls="--", lw=0.5)
		plt.tight_layout()
		plt.show()

		# Print summary
		if verbose:
			print("\n=== Distribution Fit Summary ===")
			for name, res in fitted_results.items():
				if 'aic' in res:
					print(f"{name:16} | AIC: {res['aic']:.2f} | Params: {res['num_params']} | LogL: {res['log_likelihood']:.2f}")

		return fitted_results

	def compare_model_with_trade_data(self, csv_path, csv_col, pred_col="estimated_imports", year=None, raw_log=True, pred_log=True, rm_outliers=False):
		"""
		Compare model predictions with raw trade data by creating a scatter plot.
		
		Parameters
		----------
		csv_path : str
			Path to CSV file containing raw trade data
		csv_col : str
			Column name in csv_path containing the observed trade values
		pred_col : str, optional
			Column name containing model predictions, defaults to "estimated_imports"
		year : int, optional
			Year to filter the data, defaults to 2000
		raw_log : bool, optional
			Whether to log-transform the raw data axis, defaults to True
		pred_log : bool, optional
			Whether to log-transform the predicted data axis, defaults to True
		rm_outliers : bool, optional
			Whether to remove outliers from the plot, defaults to False
		"""
		df_raw = pd.read_csv(csv_path)
		if year:
			df_raw = df_raw[df_raw["year"] == year]
		
		df_merged = df_raw.merge(self.predicted_marginal_df, on="ISO3", how="outer")
		self._limplot(df_merged, x_col=csv_col, y_col=pred_col, x_log=raw_log, y_log=pred_log, rm_outliers=rm_outliers)

	## Overrides ##

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
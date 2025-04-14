import xarray as xr
import networkx as nx
import numpy as np

'''
netcdf_path is a dataset with lat lon variables (no time)
outflow_var is the data_var that stores out-flow data
inflow_var is the data_var that stores in-flow data
'''
class FlowNetwork:
	def __init__(self, netcdf_path, outflow_var, inflow_var, thresh=0, time=None):
		self.ensemble = nx.DiGraph()
		self.graph = nx.DiGraph()
		self.ds = xr.load_dataset(netcdf_path)
		if time:
			self.ds = self.ds.sel(time = time)
		self.inflow_var = inflow_var
		self.outflow_var = outflow_var
		self.thresh = thresh
		self.distances = xr.load_dataset('distances.nc')

	'''
	Main driver for the network generation. First generates the ensemble probability distribution and then samples the graph
	'''
	def run_model(self, ensemble_model = 'gravity', graph_model = 'config'):
		self.generate_ensemble(ensemble_model)
		self.sample_ensemble(graph_model)

	'''
	Generates the ensemble
	'''
	def generate_ensemble(self, model):
		if model == 'gravity':
			self.gravity_model(self)


	'''
	Samples the edges of the graph based on the existing ensebmle
	'''
	def sample_ensemble(self, model):
		if self.
		if model == 'config':
			self.config(self)

	# returns the distance between the pair of coordinates
	def dist(self, coord1, coord2):
		# Convert degrees to radians
		lat1, lon1 = np.radians(coord1['lat']), np.radians(coord1['lon'])
		lat2, lon2 = np.radians(coord2['lat']), np.radians(coord2['lon'])
		
		angle_delta = np.arccos(
			np.sin(lat1) * np.sin(lat2) +
			np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
		)

		return angle_delta * 6371


	# Generates edges between pairs of grid cells in self.ds and stores them in self.graph, overwriting any previously generated edges
	# Edge probability is proprtional to m1 * m2 / r
	def gravity_model(self):
		self.edge_distribution.remove_edges_from(list(self.edge_distribution.edges))

		in_data = self.ds.where(self.ds[self.inflow_var] > self.thresh, drop=True)
		in_coords = [self.ds.sel(lat=lat,lon=lon).coords for lat,lon in zip(in_data.lat.values, in_data.lon.values)]
		
		out_data = self.ds.where(self.ds[self.outflow_var] > self.thresh, drop=True)
		out_coords = [self.ds.sel(lat=lat,lon=lon).coords for lat,lon in zip(out_data.lat.values, out_data.lon.values)]
		
		normalization_constant = 1 / (len(out_coords) * len(in_coords))
		for in_coord in in_coords:
			for out_coord in out_coords:
				self.edge_distribution.add_edge(in_coord, out_coord, weight=normalization_constant * self.ds.sel(in_coord)[self.inflow_var] * self.ds.sel(out_coord)[self.outflow_var])

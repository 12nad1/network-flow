import xarray as xr
import networkx as nx

'''
netcdf_path is a dataset with lat lon variables (no time)
outflow_var is the data_var that stores out-flow data
inflow_var is the data_var that stores in-flow data
'''
class FlowNetwork:
	def __init__(self, netcdf_path, outflow_var, inflow_var, thresh=0, time=None):
		self.edge_distribution = nx.DiGraph()
		self.graph = nx.DiGraph()
		self.ds = xr.load_dataset(netcdf_path)
		if time:
			self.ds = self.ds.sel(time = time)
		self.inflow_var = inflow_var
		self.outflow_var = outflow_var
		self.thresh = thresh
		self.distances = xr.load_dataset('distances.nc')

	def generate_weighted_edges(self, model):
		if model == 'gravity':
			self.gravity_model(self)
			
	# returns the distance between the pair of coordinates
	def dist(self, coord1, coord2):
		return self.distances.sel(coord1['lat'], coord2['lat'], coord1['lon'], coord2['lon'])

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
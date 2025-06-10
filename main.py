from FlowNetwork import FlowNetwork
import os

verbose = True

# Initialize the flow network with NetCDF data and variable names
grid_data_path = os.path.join('data', 'L.T.iron_flows.2000-2016_v2.a.nc')
trade_csv_path = os.path.join('data', 'iron_raw_import_export.csv')
json_path = os.path.join('data', 'grouped_region.json')
bilateral_csv_path = os.path.join('data', 'iron_io_stage_2_v2.csv')
trade_tariff_path = os.path.join('data', 'tariffsPairs_88_21_vbeta1-2024-12.csv')

fn = FlowNetwork(grid_data_path, 'source_1', 'sink_2', time='2016')
# fn.gravity_model(distance='pairwise_haversine', threshold_percentile=90, year=2016, verbose=True)
fn.gravity_model(distance='tariff', threshold_percentile=100, trade_tariff_path=trade_tariff_path, year=2016, mode='exponential', tariff_weight_factor=2.0, verbose=True)
fn.ipf_flows(max_iters=100, tol=1e-6, verbose=verbose)
df_2 = fn.bilateral_flow(2016, json_path)
fn.validate_bilateral(bilateral_csv_path, year=2016, x_log=True, y_log=True, rm_outliers=False)
fn.plot_network_map(caption=None, num_edges=500, vmin=0, vmax=1e6, extend_max=True, extend_min=True, color='coolwarm', levels=10, edge_thickness=1, edge_alpha=0.5, edge_color='gray')

fn.save_data('data/saved_flow_network', verbose=False)
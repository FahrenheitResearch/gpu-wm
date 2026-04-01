This folder is a lightweight staging area for the most useful run artifacts.

Use `python tools/collect_node_plots.py` to refresh it from the latest local and node-backed runs.

Tracked here:
- this README

Ignored here:
- generated PNG weather panels and collages

Typical subfolders:
- `local_latest`
- `remote_latest`
- `remote2_latest`
- `six_hour_reference`
- `panhandles_hrrr_latest_local`
- `panhandles_hrrr_latest_remote`
- `panhandles_hrrr_compare`

Current notable artifact:
- `panhandles_hrrr_compare/panhandles_hrrr_compare_t00003600.png`
  - GPU-WM vs HRRR `+1 h` side-by-side for the 4 km Panhandles realism benchmark
  - fields: 2 m temperature, 2 m relative humidity, 10 m wind speed, composite reflectivity

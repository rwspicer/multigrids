# Changelog
All notable changes to this project will be documented in this file.

## [0.6.2] - 2021-02-08
### Added
- Multigrid.clip_polygon_raster that clips to a shape in a vector file
- Multigrid.clip_raster_translate that clips to extend using gdal translate

### Updated
- Cleanup documentation
- Cleanup imports
- tools.tiffs_to_array creates a memory mapped array 
- tools.tiffs_to_array  changed messages provided if verbose is true 

## fixed
- Multigrid.clip_raster fixes way raster.zoom_box is called
- Multigrids filters feature is fixed

## [0.6.1] - 2021-01-26
### Added
- clip_grids implementation

### Updated
- docs for zoom_to


## [0.6.0] - 2021-01-04
### Added
- filters function

## [0.5.1] - 2020-12-10
### Added
- function for temporal girds to reset grid name map

### Fixed
- save all figures title bugfix

## [0.5.0] - 2020-12-02
### Added
- functions to find location of maximum values

### Changed
- zoom_to functionality preserves/updates raster metadata for zoom view

## [0.4.0] - 2020-11-24
### Changed
- refactored code to use spicebox as a dependency  and to clean up copy and 
  pasted code
- Multigrid.save_as_geotiff: uses spicebox.raster.save_raster 

## [0.3.0] - 2020-04-30
### Added
- auto crate MultiGrid functionality (create_multigrid.py)
- new dict format option for raster_metadata in MultiGrid config
- create subset functions
### Changed
- default figure color bar has been modified
### Fixed
- TempoalMultiGrid dataset_name bug

## [0.2.1] - 2019-10-12
### added 
- clip generation to TemporalMultiGrid
- clip generation to TemporalGrid
- descriptions and dataset_names to grids created by atm.grids.grids.ModelGrids


## [0.2.0] - 2019-05-10
### added
- changelog

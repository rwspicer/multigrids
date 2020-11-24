# Changelog
All notable changes to this project will be documented in this file.

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

# Virtual Pool Coach

Virtual Pool Coach system built during the Huawei Hackaton 2023 to detect balls and pockets on a pool table, determine the best shot to play, and projecting shots' trajectories and ball locations on the playing table. Projection and Table Detection need to be calibrated beforehand.

- controller: Controller module for the whole software
- game_model: TBA, contains several modules defining the game model for the software
- notebooks: Contains notebooks and code for development, these are not part of the final software
- projection: Contains modules for projecting lines and circles on the pool table, as well as camera calibration and transformations
- strategy: Contains modules for calculating the optimal strategy (handles multiple scenarios) and drawing lines/balls on the imaginary table
- vision: Contains modules with several methods for ball detection, table detection, and table transformation
- main.py: Main script of the software
- simulate_strategy.py: Simulation of a strategy for testing and debugging purposes

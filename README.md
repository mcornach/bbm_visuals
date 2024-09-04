# Bid-Based Modeling LDRD Visualizations

This repository has code for making plots for the bid-based modeling LDRD project.
Code here is to be run after generating saved data from scheduler.py 
(from the espa_comp repo) under different conditions.

## Generating Plots

To set directories, make edits directly in plotting_tools.py.
One directory must be named 'bbm_baseline' and the others
named 'bbm_options_[N]" (where [N] is 1, 2, 3, etc.).

The surplus plotter will generate a box and whisker plot of
each option's surplus relative to the baseline surplus. You have
options to aggregated daily (default), hourly, or use whatever
period is in the raw data.

The revenue plotter will generate three box and whisker plots.

- Income: the income plot shows the storage unit average revenue (daily or hourly) compared across all options.
- Degradation: this shows the storage unit average degradation across all options
- Revenue: this gives the Income - Degradation for storage units across all options

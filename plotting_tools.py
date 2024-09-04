#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for making different plots for the bid-based LDRD project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

class SurplusPlotter:
    
    def __init__(self, dirlist, mkt_fname='market_summary_TSRTM.csv', sample_interval='day'):
        """ Initialize with a list of saved directories to use for plotting
        Each directory may contain several market surplus files
        Requires the same files in each directory
        """
        # Load inputs into class variables
        if type(dirlist) == str: # Handling for a single directory input
            dirlist = [dirlist]
        self.find_surplus_types(dirlist[0])
        self.dirlist = dirlist
        self.mkt_fname = mkt_fname
        assert sample_interval in ['day', 'hour', None]
        if 'day' in sample_interval.lower():
            self.si = "D"
        elif 'hour' in sample_interval.lower():
            self.si = 'H'
        else:
            self.si = None
        # None defaults for class attributes
        self.surplus_types = None
        self.timeseries = None
        self.baseline_surplus = None
        self.option_surplus = None
        self.surplus_diff = None
        # Load data
        self.get_differences()
        
    def find_surplus_types(self, firstdir):
        """ Checks for the different surplus types and loads them into a list """
        surplus_files = glob.glob(os.path.join(firstdir, 'market_summary*csv'))
        self.surplus_types = ['.'.join(fname.split('.')[:-1]) for fname in surplus_files]
        self.surplus_types = [fname.split('_')[-1] for fname in self.surplus_types]

    def get_baseline_surplus(self, baseline_dir='bbm_baseline'):
        """ Loads the baseline physical surplus and timeseries """
        root = os.path.split(self.dirlist[0])[0]
        surplus_df = pd.read_csv(os.path.join(root, baseline_dir, self.mkt_fname), index_col='time')
        surplus_df.index = pd.to_datetime(surplus_df.index, format='%Y%m%d%H%M')
        # Resample
        if self.si is not None:
            surplus_df = surplus_df.resample(self.si).sum()
        # Timeseries and surplus
        self.timeseries = surplus_df.index.values
        self.baseline_surplus = surplus_df['surplus_phys'].values

    def get_option_surplus(self):
        """ Saves a dictionary to the object with the surplus for different options """
        self.option_surplus = dict()
        for directory in self.dirlist:
            if 'option' in directory:
                key = '_'.join(directory.split('_')[1:])  # converts bbm_option_N -> option_N

                df = pd.read_csv(os.path.join(directory, self.mkt_fname), index_col='time')
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')
                # Resample
                if self.si is not None:
                    df = df.resample(self.si).sum()
                self.option_surplus[key] = df['surplus_phys'].values

    def get_differences(self):
        """ Loads baseline and options and makes a dictionary of the differences """
        # First load baseline and options surplus
        self.get_baseline_surplus()
        self.get_option_surplus()
        # Now compute differences
        self.surplus_diff = dict()
        for key, opt_surplus in self.option_surplus.items():
            self.surplus_diff[key] = self.baseline_surplus - opt_surplus

    def plot_differences(self):
        """ Creates a box and whisker plot of the surplus differences"""
        # Ensure the differences have been calculated
        if self.surplus_diff is None:
            self.get_differences()
        # Configure styles
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        fig = plt.figure()
        ax = plt.gca()
        for option, data in self.surplus_diff.items():
            sns.boxplot(y=data)
        os.makedirs('figs', exist_ok=True)
        plt.savefig('figs/surplus_bw.png', bbox_inches='tight')
        plt.show()

class RevenuePlotter:

    def __init__(self):
        pass

if __name__ == '__main__':
    id_str = 'bbm'
    # Gets all directories in scenarios folder with 'bbm' in name
    dirs = glob.glob(os.path.join('../scenarios', f'{id_str}*'))
    dirs = sorted(dirs)
    # Convert to absolute paths
    dirs = [os.path.abspath(d) for d in dirs]
    dirs = ['../scenarios/bbm_baseline', '../scenarios/bbm_option_3']
    splotter = SurplusPlotter(dirs)
    splotter.plot_differences()
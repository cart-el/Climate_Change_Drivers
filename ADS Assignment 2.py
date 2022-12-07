# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 06:39:16 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tool (data):
    data_dir = "C:/Users/USER/Desktop/Data Science Course/Applied Data Science 1/ASS 3/"
    file = data_dir + data
    WB_data = pd.read_excel(file, skiprows=3)
    return WB_data, WB_data.transpose()
       
Climate_change = tool("WD_data.xls")
#print(Climate_change)
        
doc = Climate_change[0]
#print(doc)
scope = doc[doc['Country Name'].isin(['Sub-Saharan Africa', 
                                      'Middle East & North Africa',
                                      'Europe & Central Asia', 
                                      'North America', 
                                      'South Asia',
                                      'East Asia & Pacific',
                                      'Latin America & Caribbean'])]
#print(scope)

CO2_emission = scope[(scope['Indicator Name'] == 'CO2 emissions (kt)')]
CO2_emission = CO2_emission.round(2)
#print(CO2_emission)
CO2_emission = CO2_emission.iloc[:, 4:]
#print(CO2_emission)
CO2_emission = CO2_emission.dropna(axis=1)
#print(CO2_emission)
CO2_emission = CO2_emission.transpose()
#print(CO2_emission)
CO2_emission = CO2_emission.reset_index(inplace=False)
# CO2_emission = pd.DataFrame(CO2_emission)
# print(CO2_emission)
CO2_emission = CO2_emission.set_axis(["Year",
                      "East Asia & Pacific", 
                      "Europe & Central Asia", 
                      "Latin America & Caribbean", 
                      "Middle East & North Africa", 
                      "North America", 
                      "South Asia", 
                      "Sub-Saharan Africa"], axis=1)

print(CO2_emission)

def lineplot(df, x, columns, xlabel, ylabel, title, file):
    """
    Produces a line plot from a dataframe. x-limits are adjusted to remove
    empty spaces at the edges.
    df: name of the dataframe
    x: 1D array or dataseries with the x-values
    columns: names of the columns in df to be plotted. Also used for the 
            legend.
    xlabel, ylabels: labels for x and y axis.
    title: title of plot
    file: file name to store the plot as png file
    """

    plt.figure()

    # loop over the columns
    for c in columns:
        plt.plot(x, df[c], label=c)
    
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # set x-limits
    plt.xlim(min(x), max(x))
    plt.xticks(rotation=90)
    #ax.set_xticks(np.arange(0, len(x)+1, 5))
    
    plt.savefig(file) 
    plt.show()

         
columns = ["East Asia & Pacific", "Europe & Central Asia",
           "Latin America & Caribbean", "Middle East & North Africa",
           "North America", "South Asia", "Sub-Saharan Africa"]

x = CO2_emission['Year']
x_label = "Year"
y_label = "CO2 Emission"
title = "Analysis Of CO2 Emission"
file = "CO2_Emission Lineplot"

lineplot(CO2_emission, x, columns, x_label, y_label, title, file)



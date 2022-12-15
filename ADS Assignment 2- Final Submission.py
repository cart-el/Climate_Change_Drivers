# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 06:39:16 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def f_read (data):
    """
    Reads a file from the specified directory in world bank format and 
    outputs two dataframes: one is main data while the second is the 
    transposed version of the same data. 
    data: name of file in directory
   
    Returns
    -------
     Dataframe: A tuple

    """
    data_dir = "C:/Users/USER/Desktop/Data Science Course/Applied Data Science 1/ASS 3/"
    file = data_dir + data
    WB_data = pd.read_excel(file, skiprows=3)
    return WB_data, WB_data.transpose()
       
Climate_change = f_read("WD_data.xls")  #read file using function 
#print(Climate_change)
 
#the function returns two dataframes and they can be indexed using [0] and [1]
#with [0] being the original data and [1] being the transposed version       
file1 = Climate_change[0]  #indexing the tuple returned from the function-call
#print(file1)

#Slicing the dataframe to extract rows of interest
scope1 = file1[file1['Country Name'].isin(['Sub-Saharan Africa', 
                                      'Middle East & North Africa',
                                      'Europe & Central Asia', 
                                      'North America', 
                                      'South Asia',
                                      'East Asia & Pacific',
                                      'Latin America & Caribbean'])]
#print(scope)
#indexing down the rows to extract attribute of interest
#The aim is to filter out the values for CO2 emission for the regions
#between 1990 and 2019 and ploting a time series for this.
CO2_emission = scope1[(scope1['Indicator Name'] == 'CO2 emissions (kt)')]
CO2_emission = CO2_emission.round(2) #rounding off to 2dp
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

#defining the function variables         
columns = ["East Asia & Pacific", "Europe & Central Asia",
            "Latin America & Caribbean", "Middle East & North Africa",
            "North America", "South Asia", "Sub-Saharan Africa"]
x = CO2_emission['Year']

#labelling the plot
x_label = "Year"
y_label = "CO2 Emission"
title = "Analysis Of CO2 Emission"
file = "CO2_Emission Lineplot"

#initiating the plot with the function 
lineplot(CO2_emission, x, columns, x_label, y_label, title, file)


# ##############BAR PLOT#########################
#The aim is to produce a compound barplot of electricity consumption from
#various sources within the regions of interest and plotting a line plot 
#of renewable energy on same grapgh to have a sense of how much of renewable
#energy is being consumed when compared to ther traditional sources of

#Tthe transposed version is called and the attributes of interest are indexed
#from the rows & columns.
file2 = Climate_change[1]  
print(file2)
elect_EAP = file2.iloc[55:64, [4788, 4841, 4844, 4845, 4846, 4847]]
elect_EAP = elect_EAP.dropna(axis=0)
elect_EAP = elect_EAP.reset_index(inplace=False)
elect_EAP = elect_EAP.set_axis(["Year",
                                "Urban population (% of total population)",
                                "Electricity production from renewable sources, excluding hydroelectric (% of total)",
                                "Electricity production from oil sources (% of total)",
                                "Electricity production from nuclear sources (% of total)",
                                "Electricity production from natural gas sources (% of total)",
                                "Electricity production from hydroelectric sources (% of total)"], 
                                axis=1, inplace=False)

print(elect_EAP)

def bar_plot(year, arr, **kwargs):  #define line plot function
   
    '''
    Creates a bar chart

    Parameters
    ----------
    arr : float or array-like
        The x coordinates of bars.
    
    Returns
    -------
    fig : bar container
        Container with all the bars.

    '''
  
    #plt.subplots(figsize = (10, 6))  #define the plot environment
    b_width = w
    
    #define the position of the bars
    br1 = np.arange(len(arr))  
    br2 = [x + b_width for x in br1]  
    br3 = [x + b_width for x in br2]
    br4 = [x + b_width for x in br3]  
    #br5 = [x + b_width for x in br4]
    #br6 = [x + b_width for x in br5]  
    #br7 = [x + b_width for x in br6]
    
    #define plot parameters 
    plt.bar(br1, arr1, color="darkslategray", width=b_width, edgecolor="grey", 
            label=label1)
    plt.bar(br2, arr2, color='limegreen', width=b_width, edgecolor="grey", 
            label=label2)
    plt.bar(br3, arr3, color='crimson', width=b_width, edgecolor="grey", 
            label=label3)
    plt.bar(br4, arr4, color="c", width=b_width, edgecolor="grey", 
            label=label4)
    #plt.bar(br5, arr5, color="orange", width=b_width, edgecolor="grey", 
    #        label=label5)
    #plt.bar(br6, arr6, color="brown", width=b_width, edgecolor="grey", 
      #       label=label6)
    plt.xticks([t + w + 0.15 for t in range(len(arr))], [2011, 2012, 2013, 
                                                         2014, 2015])
    plt.plot()
    plt.legend()
    #plt.show()
    #return

    
#Initialise the function variables
w = 0.2 
arr1 = elect_EAP["Electricity production from oil sources (% of total)"]
arr2 = elect_EAP["Electricity production from nuclear sources (% of total)"]
arr3 = elect_EAP["Electricity production from natural gas sources (% of total)"]
arr4 = elect_EAP["Electricity production from hydroelectric sources (% of total)"] 
arr5 = elect_EAP["Electricity production from renewable sources, excluding hydroelectric (% of total)"]

year = elect_EAP["Year"]
label1 = "Oil sources"
label2 = "Nuclear sources"
label3 = "Natural gas sources"
label4 = "Hydroelectric sources"
label5 = "Renewable sources, excluding hydro"

plt.title("Analysis of Electricity Production in East Asia & Pacific, 2011-2015")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Rate of Change (% of total)", fontsize=12)
#plt.legend(fontsize=12)
plt.ylim(0,16)
#plt.style.use('seaborn') 

#implementing the bar_plot function
bar_plot(year, arr1)
#line plot for renewable energy consumtion
plt.plot(year, arr5, label=label5, color='black', marker='o')
plt.legend(loc='lower right', bbox_to_anchor=(1.05, -0.42), ncol=2)

plt.savefig('Regional Electricity Consumption')
plt.show()



# #####################BUBBLE PLOT##################
#The aim is to produce a bubble plot of life expectancy against GDP/Capita
#while having the regional population as the sizing parameter. This would
#give a sense of the extent to which climate change affects the lives of 
#inhabitants of the regions between 1968 and 2020. In order to achieve this,
#Life expectancy data and GDP/Capita data were fetched from the world bank
#website.

#The original data version is called and the attributes of interest are indexed
#from the rows & columns.

pop = file1.iloc[[4791, 4943, 10187, 11631, 12923, 15507, 16495], [0,12,64]]
pop = pop.rename(columns = {'Country Name':'REGION', '1968':'POPULATION(1968)',
                            '2020':'POPULATION(2020)'}, inplace=False)
#print(pop)

#Using the filereader function call to pull the Life expectancy data
WB_data2 = f_read("Life_Expectancy.xlsx")
WB_data2 = WB_data2[0]
Life_Exp = WB_data2.iloc[[63,65,134,153,170,204,217], [0,12,64]]
Life_Exp = Life_Exp.rename(columns = {'Country Name':'REGION', '1968':
                                      'LIFE EXPECTANCY(1968)',
                                      '2020':'LIFE EXPECTANCY(2020)'}, 
                           inplace=False)
#print(Life_Exp)

#Using the filereader function call to pull the GDP/Capita data
WB_data3 = f_read("GDP Per Capita.xlsx")
WB_data3 = WB_data3[0] 
GDP_pc = WB_data3.iloc[[63,65,134,153,170,204,217], [0,12,64]]
GDP_pc = GDP_pc.rename(columns = {'Country Name':'REGION', '1968':
                                  'GDP/CAPITA(1968)',
                                  '2020':'GDP/CAPITA(2020)'}, inplace=False)
#print(GDP_pc)


#Defining the bubble plot variables, with x0, y0 & s0 being the dimensional 
#parameters for the 1968 plot and x1, y1 & s1 being the dimensional 
#parameters for the 2020 plot.
x0 = GDP_pc['GDP/CAPITA(1968)']
y0 = Life_Exp['LIFE EXPECTANCY(1968)']
s0 = pop['POPULATION(1968)']/900000
x1 = GDP_pc['GDP/CAPITA(2020)']
y1 = Life_Exp['LIFE EXPECTANCY(2020)']
s1 = pop['POPULATION(2020)']/900000
c = ["blue", "red","green", "cyan", "orange", "purple", "yellow"]
label = ["East Asia & Pacific","Europe & Central Asia",
         "Latin America & Caribbean","Middle East & North Africa",
         "North America","South Asia","Sub-Saharan Africa"]

#First subplot
plt.subplot(2,1,1)
bubble0 = plt.scatter(x0, y0, s0, alpha=0.8, c=c, label=label)
plt.title("Life Expectancy & GDP/Capita($)-1986")
plt.xlabel("GDP/Capita ($)")
plt.ylabel("Life Expectancy (Years)", fontsize=10)
#plt.legend(fontsize=12)
plt.ylim(40,80)
plt.xlim(0,5000)
plt.savefig('Bubble Plot- 1986')
plt.show()

#Second subplot
plt.subplot(2,1,1)
bubble1 = plt.scatter(x1, y1, s1, alpha=0.8, c=c, label=label)
plt.title("Life Expectancy & GDP/Capita($)-2020")
plt.xlabel("GDP/Capita ($)")
plt.ylabel("Life Expectancy (Years)", fontsize=10)
#plt.legend(fontsize=12)
plt.ylim(40,100)
plt.xlim(0,62000)
plt.savefig('Bubble Plot- 2020')
plt.show()




# ####################HEAT MAP#####################################
#The aim is to produce a heat map to show any form of correlation or
#otherwise among some selected climate change attributes.
#To achieve this, the correlation function will be called and this
#normalises all values of the data before the heatmap is ploted.

#The transposed data version is called and the attributes of interest are 
#indexed from the rows columns.

SA_anal = file2.iloc[34:50, [15507, 15509, 15512, 15513, 15514, 15515, 15538, 
                              15545, 15567]]
SA_anal = SA_anal.set_axis(["Population","Poverty Headcount Ratio",
                            "Mortality Rate", "Primary Completetion Rate",
                            "School Enrolment(GPI)", 
                            "Agriculture/Forestry/Fishing",
                            "Total Greenhouse Emission", 
                            "CO2 Emissions (per capita)",
                            "Cereal Yield(per ha)"], axis=1, inplace=False)

#the NANs are filled with zero since they're just a few
SA_anal["Poverty Headcount Ratio"] = SA_anal["Poverty Headcount Ratio"].fillna(0)
SA_anal = SA_anal.astype(float)  #conversion of the dataframe to a float
#print(SA_anal)

SA_cor = SA_anal.corr().round(2)
#print(SA_cor)

#plotting the heatmap and specifying the plot parameters
plt.imshow(SA_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(SA_cor)), SA_cor.columns, rotation=90)
plt.yticks(range(len(SA_cor)), SA_cor.columns)
plt.gcf().set_size_inches(8,5)

#labelling of the little boxes and creation of a legend
labels = SA_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x,y, '{:.2f}'.format(labels[y,x]), ha='center', va='center',
                  color='black')
plt.title('Correlation Map of South Asia Region')
plt.savefig("Heat Map of SA Region" )




#######LINE CHART (POVERTY)#############
#The aim is to produce a line plot to show the poverty headcount ratio
#of the regions and compare them with the world average so as to give an
#visual representation of ill-performing regions with a quick glance.
#otherwise among some selected climate change attributes.

#the transposed data version is called and the attributes of interest are 
#indexed from the rows columns.

poverty_hc = file2.iloc[[34,47,56,62], [4793,4945,10189,11633,15509,
                                        16497,19689]]
poverty_hc = poverty_hc.dropna()
poverty_hc = poverty_hc.reset_index(inplace=False)
poverty_hc = poverty_hc.set_axis(["Year",
                          "East Asia $ Pacific",
                          "Europe & Central Asia",
                          "Latin America & Caribbean",
                          "Middle East & North Africa",
                          "South Asia",
                          "Sub-Saharan Africa",
                          "World Average"], axis=1, inplace=False)
print(poverty_hc)

#intiating the plot and attributing unique characters to aid easy visual 
#assimilation
plt.figure()
x = poverty_hc['Year']
plt.plot(x, poverty_hc['East Asia $ Pacific'], linestyle='dashed', 
          label="EAP")
plt.plot(x, poverty_hc['Europe & Central Asia'], linestyle='dashed',
          label="ECA")
plt.plot(x, poverty_hc['Latin America & Caribbean'], linestyle='dashed',
          label="LAC")
plt.plot(x, poverty_hc['Middle East & North Africa'], linestyle='dashed',
          label="MENA")
plt.plot(x, poverty_hc['South Asia'], linestyle='dashed', label="SA")
plt.plot(x, poverty_hc['Sub-Saharan Africa'], linestyle='dashed',
          label="SSA")
plt.plot(x, poverty_hc['World Average'], marker="o", label="World Avg",
          color="black", linewidth=2)

#adding labels and title to the plot
plt.xlabel("Year")
plt.ylabel("Poverty Headcount (% of population)")
plt.title("Poverty Headcount Ratio Across Regions")
plt.legend(loc='upper right', ncol=2)
plt.savefig('Poverty Headcount') 
plt.show()



# # ####################BOX PLOT########################
#The aim is to produce a box plot to show at a quick glance the 
#dispersion, skewness and median of the poverty headcount ratio withing
#the selected regions while also checking for any outliers. 

#the transposed data version is called and the attributes of interest are 
#indexed from the rows columns.
poverty_hc_SS = file2.iloc[34:64, [4793,15509,11633,16497]]
#dropping all NANs as it corrupts the data
poverty_hc_SS = poverty_hc_SS.dropna()  
poverty_hc_SS = poverty_hc_SS.reset_index(inplace=False)
poverty_hc_SS = poverty_hc_SS.set_axis(["Year", "East Asia $ Pacific",
                                        "South Asia",
                                        "Middle East & North Africa",
                                        "Sub-Saharan Africa"],
                                        axis=1, inplace=False)
#print(poverty_hc_SS)


# list of names
Regions = ["EAP", "SA", "MENA", "SSA"]

data = [poverty_hc_SS["East Asia $ Pacific"], poverty_hc_SS["South Asia"], 
        poverty_hc_SS["Middle East & North Africa"], 
        poverty_hc_SS["Sub-Saharan Africa"]]

#plotting the data
plt.figure()
plt.boxplot(data, labels=Regions)

#labeling the plot
plt.ylabel("Poverty Headcount Ratio")
plt.title("Box Plot of Selected Regional Poverty Ratio")
plt.legend()
plt.savefig("box.png")
plt.show()



#########Statistical Analysis#########
#The aim is to expand on the statistical data gathered from the boxplot
#This gives a table of the statistical properties of the plotted data
#and the .describe() function is used to achieve this
poverty_hc_stat = poverty_hc_SS[['East Asia $ Pacific', 'South Asia', 
                                'Middle East & North Africa', 
                                'Sub-Saharan Africa']] = poverty_hc_SS[[
                                    'East Asia $ Pacific', 'South Asia', 
                                    'Middle East & North Africa', 
                                    'Sub-Saharan Africa']].astype(int)
poverty_hc_stat = poverty_hc_stat.describe()
print(poverty_hc_stat)




#####################AGRICULTURAL BAR PLOT##########################
#The aim is to produce a compound barplot of the contribution of Agriculture
#to GDP in the regions of interest. This is meant to reveal the regions whose 
#means of livelihood will be affected the most by climate change.

#Tthe original version is called and the attributes of interest are indexed
#from the rows & columns.

Agric_GDP = file1.iloc[[4799,4951,10195,11639,12931,15515,16503], 
                       [41,48,55,64]]
Agric_GDP = Agric_GDP.set_axis(["EAP", "ECA", "LAC", "MENA", "NA", "SA", 
                                "SSA"], axis=0, inplace=False)
#print(Agric_GDP)

#plotting the bar chart
Agric_GDP.plot(kind='bar')

#labeling and title
plt.title('GDP Contribution of Agriculture/Fishing/Forestry Across Regions')
plt.xlabel('Regions')
plt.ylabel('% of GDP')
plt.savefig('GDP Agriculture') 
plt.show()
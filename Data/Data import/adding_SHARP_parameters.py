import pandas as pd 
import drms
import numpy as np 
import time 
from datetime import datetime

X = pd.read_csv("dataset.csv").reset_index()
c = drms.Client() 

## Get the active region index in terms of HARPNUM
harpnum_indexes = dict()
with open("all_harps_with_noaa_ars.txt") as f: 
    lines = f.readlines()
    for line in lines[1:]: 
        HARPNUM, NOAA_ARS = line.split()
        NOAA_ARS = NOAA_ARS.split(",")
        for value in NOAA_ARS: 
            harpnum_indexes[value] = HARPNUM
            
# Extract SHARP paramters as a query 

str_1 = str('''           USFLUX  Total unsigned flux in Maxwells
           ERRVF   Error in the total unsigned flux
           CMASK   Number of pixels used in the USFLUX calculation
           MEANGAM Mean inclination angle, gamma, in degrees
           ERRGAM  Error in the mean inclination angle
           MEANGBT Mean value of the total field gradient, in Gauss/Mm
           ERRBT   Error in the mean value of the total field gradient
           MEANGBZ Mean value of the vertical field gradient, in Gauss/Mm
           ERRBZ   Error in the mean value of the vertical field gradient 
           MEANGBH Mean value of the horizontal field gradient, in Gauss/Mm
           ERRBH   Error in the mean value of the horizontal field gradient
           MEANJZD Mean vertical current density, in mA/m2
           ERRJZ   Error in the mean vertical current density
           TOTUSJZ Total unsigned vertical current, in Amperes
           ERRUSI  Error in the total unsigned vertical current
           MEANALP Mean twist parameter, alpha, in 1/Mm
           ERRALP  Error in the mean twist parameter
           MEANJZH Mean current helicity in G2/m
           ERRMIH  Error in the mean current helicity
           TOTUSJH Total unsigned current helicity in G2/m
           ERRTUI  Error in the total unsigned current helicity
           ABSNJZH Absolute value of the net current helicity in G2/m
           ERRTAI  Error in the absolute value of the net current helicity
           SAVNCPP Sum of the absolute value of the net current per polarity in Amperes
           ERRJHT  Error in the sum of the absolute value of the net current per polarity
           MEANPOT Mean photospheric excess magnetic energy density in ergs per cubic centimeter
           ERRMPOT Error in the mean photospheric excess magnetic energy density
           TOTPOT  Total photospheric magnetic energy density in ergs per centimeter
           ERRTPOT Error in the total photospheric magnetic energy density
           MEANSHR Mean shear angle (measured using Btotal) in degrees
           ERRMSHA Error in the mean shear angle
           SHRGT45 Area with shear angle greater than 45 degrees (as a percent of total area)
           R_VALUE Flux along gradient-weighted neutral-line length in Maxwells
           ''')
SHARP_names = [c for c in str_1.split() if c.isupper()][1:]
SHARP_query = ""
for i in SHARP_names:
    SHARP_query += (i + ",") 

# Define query function
def query_drms(row, nhours): 
    
    try:
        index = str(int(X["noaa_ar"].iloc[row]))
        date_time_str = X["event_peaktime"].iloc[row]
        date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        time = str(date_time - pd.DateOffset(hours=nhours))
        keys = c.query(f'hmi.sharp_cea_720s[{harpnum_indexes[index]}][{time}][]', key= SHARP_query)
    except:
        keys = pd.DataFrame([[None for i in range(len(SHARP_names))]])
    merged_row = pd.concat((X.iloc[[row]].reset_index(), keys), axis = 1, ignore_index = True)
    return merged_row 


# How many hours you want to go back for SHARP data 
nhours = 24

# Query through entire dataset for SHARP data 
SHARP_data = pd.DataFrame()
for n_row in range(X.shape[0]): 
    SHARP_data = pd.concat((SHARP_data, query_drms(n_row,nhours)),ignore_index = True)

SHARP_data = SHARP_data.drop([0], axis = 1)
SHARP_data_colnames = []
[SHARP_data_colnames.append(i) for i in X.columns]
[SHARP_data_colnames.append(i) for i in SHARP_names]
SHARP_data.columns = SHARP_data_colnames
SHARP_data.to_csv(f"SHARP_data_{nhours}.csv", index_label=False) 
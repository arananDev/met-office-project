import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

"""
This script takes a GOES flare list that has been created using `swpc_report_flarelist.py` and an AR list
created using `get_ar_data.py` and merges them to create a flare list that then includes AR data associated with each flare.
You can also merge it the other way if you like to create an AR database (for each day) with associated flares. 

It outputs a csv file of the merged flare and AR propery list.
"""


## read in flarelist
flare_list = pd.read_csv("swpc_event_list.csv")
# create a column "matchtime" which will be used to merge with the AR data
# tidying up some other columns (i.e. make AR num a string, and sort out dates of start, peak and end of flare)
flare_list["noaa_ar_no"] = flare_list["noaa_ar"].astype(str)
flare_list["event_starttime"] = pd.to_datetime(flare_list["date"].astype(str) + flare_list["start_time"].astype(str).str.zfill(4))
flare_list["event_peaktime"] = pd.to_datetime(flare_list["date"].astype(str) + flare_list["max_time"].astype(str).str.zfill(4))
flare_list["event_endtime"] = pd.to_datetime(flare_list["date"].astype(str) + flare_list["end_time"].astype(str).str.zfill(4), errors = "coerce")
flare_list.dropna(subset = ["event_endtime"], inplace = True)
# drop unneccessary columns
flare_list["matchtime"] = flare_list["date"]
flare_list.drop(["start_time", "max_time", "end_time", "event_no", "ts", "date", "goes_sat","goes_channel"], axis=1, inplace=True)


# read in active region list
ar_data = pd.read_csv("MU_noaa_ars_plages.csv")

# create date column 
date = []
for i in range(ar_data.shape[0]):
    if ar_data["day"][i] < 10: 
        day = "0" + str(ar_data["day"][i])
    else:
        day = str(ar_data["day"][i])
    if ar_data["month"][i] < 10: 
        month = "0" + str(ar_data["month"][i])
    else:
        month = str(ar_data["month"][i])
    year = str((ar_data["year"])[i])
                
    date.append(int(year + month + day ) )
    
ar_data["matchtime"] = date   
# rename AR column so that it can be merged with GOES flare list
ar_data["noaa_ar_no"] = ar_data["noaa_ar_no"].astype(str)
ar_data.rename(columns={"date":"AR issue_date"}, inplace=True)


## merge the files!
merged_db = pd.merge(ar_data, flare_list, how="left", on=["matchtime", "noaa_ar_no"])
merged_db.to_csv("dataset.csv", index_label=False)
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:00:52 2025

@author: Aurélien
"""

import requests
import os
import time
#Extract all the GTFS between two given timestamps and save them in 'data' folder
dir_GTFS='data'
#%%
from_timestamp=1755138000# 14/08/25 04:20:00
#from_timestamp=1757729880# 13/09/25 04:18:00
#to_timestamp = 1755291600# 15/08/25 23:00:00
#to_timestamp = 1756596942# 31/08/25 01:35:42
to_timestamp = 1758075480# 17/09/25 04:22:00

prev_timestamp=from_timestamp
prev_data_timestamp=''
last_size=0


if not os.path.exists(dir_GTFS):
    os.makedirs(dir_GTFS)

for timestamp in range(from_timestamp,to_timestamp,86400):#Check every 24h
    url = "https://api.mobilitytwin.brussels/stib/gtfs?timestamp="+str(timestamp)
    
    r = requests.get(url,  headers={
    'Authorization': 'Bearer 17cafff86b7b9a9bf2efecae7b84e0cb59783c17e76173f4b48d9a837e3b87d679cdcff4b6716f9c60e527238293c2097f70489fe2e6142061f04e043f4fc1d1'
    })
    if r.status_code != 200:
        print("Error "+str(r.status_code)+" :"+r.reason)
        time.sleep(1)
    else:
        data_timestamp=r.headers['X-Data-Timestamp']
        #print(timestamp,end='')
        data_timestamp=data_timestamp.split('.')[0].replace(':', '-')
        if(data_timestamp!=prev_data_timestamp):
            data=r.content
            data_size=len(data)#TODO: I use the data size to distinguish the files. It is not perfectly accurate and should be replaced by a hash function
            
            print("Timestamp:",prev_timestamp,"to",timestamp-1,"("+str(timestamp-prev_timestamp)+")")
            print("Date:",data_timestamp)
            prev_data_timestamp=data_timestamp
            prev_timestamp=timestamp
            
            if(data_size!=last_size):
                print("New file: GTFS_"+data_timestamp+'.zip')
                print('')
                # with tempfile.NamedTemporaryFile(delete=False) as f:
                f = open(dir_GTFS+'/GTFS_'+data_timestamp+'.zip', 'wb')
                f.write(data)
                f.close()
                
                last_size=data_size
            else:
                print("Same file as before")
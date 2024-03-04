# -*- coding: utf-8 -*-

import pandas as pd
from pyedautils.geopy import get_distance_between_two_points

def get_current_station_data():
    """
    Gets current measurement data of all Meteo Swiss stations.

    Returns:
        df: pandas dataframe with station data
    """
    try:
        endpoint = "https://data.geo.admin.ch/ch.meteoschweiz.messnetz-automatisch/ch.meteoschweiz.messnetz-automatisch_de.csv"
        data = pd.read_csv(endpoint, encoding= 'unicode_escape', sep = ";")
        df = data[data['Messungen'].notna()]
           
        return df
    except:
        return pd.DataFrame()

def find_nearest_station(lat: float, long: float, altitude:float, sensor:str):
    """
    Returns station id of closest meteo swiss station to a coordinate.

    Args:
        lat (float): Latitude in decimal degrees.
        long (float): Longitude in decimal degrees.
        altitude (float): Altitude in meters above sea level.
        sensor (str): temp, globrad, relhum or rain  

    Returns:
        str: Meteo Swiss station ID
    """
    
    switzerland_lat_min = 45.67
    switzerland_lat_max = 47.92
    switzerland_long_min = 5.7
    switzerland_long_max = 10.7
    
    if(not((switzerland_lat_min <= lat <= switzerland_lat_max) and (switzerland_long_min <= long <= switzerland_long_max))):
        raise ValueError("Coordinates not in range for Swiss coordinate system")
    
    all_station_data = get_current_station_data()
    
    lats=all_station_data["Breitengrad"].astype("float")
    lons=all_station_data["Längengrad"].astype("float")
    
    distances=pd.DataFrame([get_distance_between_two_points((lat,long),(lats[i],lons[i])) for i in range(len(lats)) ])
    distances.columns = ['value']
    distances = distances.sort_values(by='value')
    distances = distances.reset_index()

    i = 0
    
    while i < len(distances):
        id = distances.loc[i,"index"]
        sensors = all_station_data.loc[id,"Messungen"]        
        found = False
        stationID = all_station_data.loc[id,"Abk."]
        stationIDString = all_station_data.loc[id,"Station"]
        stationAltitude = float(all_station_data.loc[id,"Stationshöhe m ü. M."])
        if((float(altitude) > (stationAltitude - 150.0)) and (float(altitude) < (stationAltitude + 150.0))):
            if(sensor=="temp" and ("Temperatur" in sensors)):
                print("Closest station for temp: " + str(stationIDString))
                found = True
                break
            elif(sensor=="globrad" and ("Globalstrahlung" in sensors)):
                print("Closest station for globalrad: " + str(stationIDString))
                found = True
                break
            elif(sensor=="relhum" and ("Feuchte" in sensors)):
                print("Closest station for relhum: " + str(stationIDString))
                found = True
                break
            elif(sensor=="rain" and ("Niederschlag" in sensors)):
                print("Closest station for rain: " + str(stationIDString))
                found = True
                break
            else:
                i += 1
                continue
        else:
            i += 1
            continue
    
    if not found:
        stationID = None
    
    return stationID


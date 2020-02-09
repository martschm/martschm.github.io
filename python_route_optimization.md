# Route Optimization - brute force (Python)

## Purpose
The program gets as input a csv-file with locations (longitude and latitude) and calculates the best route (based on time/distance) starting from a specified location using Google’s Distance Matrix API<br><br>

Here is an example of how the csv-file should be structured<br><br>
<img src="images/python_route_optimization_sample_csv.jpg?raw=true"/>
<br><br>

### Required packages


```python
import googlemaps as gm
from itertools import permutations
import numpy as np
import pandas as pd
```
<br>

### Read CSV-file


```python
def read_locations_csv(path, sep):
    """
    This function reads the csv with specified locations
    Input:
        path - local path to csv
        sep - seperator used in csv
    """
    
    df = pd.read_csv(path, sep)

    # check if csv is structured correctly
    assert (len(set(df.columns.tolist()) & set(["start", "longitude", "latitude", "location"])) == 4), \
           "CSV-file with locations not structured correclty!"
        
    return df
```
<br>

### All possible routes between two locations


```python
def get_intermediate_routes(df, do_round):
    """
    All possible partial routes, taking into account the starting
    and end point (if do_round equals True)
    Input:
        df - output from read_locations_csv
        do_round - boolean, wether end equals starting point
    Returns a df with all intermediate routes to be sent to google maps
    distance matrix api
    """
    
    # all possible intermediate routes, cartesian product
    int_routes = (
        df.assign(key=0)
        .merge(df.assign(key=0), on="key")
        .drop("key", axis=1)
    )
    
    # mask based on do_round; if True, then keep routes from X to starting point
    if do_round == True:
        mask = (int_routes["location_x"] != int_routes["location_y"])
    else:
        mask = (int_routes["location_x"] != int_routes["location_y"]) & (int_routes["start_y"] == 0)
    
    # apply mask and reset index on df
    int_routes = int_routes[mask].reset_index(drop=True)
    
    return int_routes
```
<br>

### Connection to Google Could Distance Matrix API

for more information: https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/distance_matrix.py


```python
def query_google_maps_distance_matrix(df, api_key, mode_transport, mode_metric):
    """
    Query google maps distance matrix api.
    Input:
        df - output of get_intermediate_routes
        api_key - string, personal google cloud api key
        mode_transport - string, either "walking", "driving" or "bicycling"
        mode_metric - string, either "duration" or "distance"
    Returns a list of durations/distances
    """
    
    # start goolge maps api connection 
    gmaps = gm.Client(key = api_key)
    
    # query google distance matrix api for all starting and end points
    result = []
    for i in range(df.shape[0]):
        start = (df["latitude_x"][i],df["longitude_x"][i])
        end   = (df["latitude_y"][i],df["longitude_y"][i])
        # for mode_metric == "duration", seconds are returned
        # for mode_metric == "distance", meters are returned
        result_tmp = gmaps.distance_matrix(start, end, mode=mode_transport)["rows"][0]["elements"][0][mode_metric]["value"]
        if mode_metric == "duration":
            result_tmp = round(result_tmp/60, 2)
        result.append(result_tmp)
        
    return result
```
<br>

### Create all possible tours


```python
def get_all_possible_tours(df_locations, do_round):
    """
    This functions returns a dataframe with all possible tours.
    One tour corresponds to a row in the dataframe.
    Input:
        df - output from read_locations_csv
        do_round - boolean, wether end equals starting point
    """

    # create a list of all possible tours
    visiting = df_locations[df_locations.start == 0]["location"].tolist()
    visiting_permutations = list(permutations(visiting, df_locations.shape[0]-1))
    starting = df_locations[df_locations.start == 1]["location"].tolist()[0]

    all_tours = []
    for visiting_tour in visiting_permutations:
        if do_round == True:
            tour = (starting,) + visiting_tour + (starting,)
        else:
            tour = (starting,) + visiting_tour
        all_tours.append(tour)

    # convert list of all tours to dataframe
    df_all_tours = pd.DataFrame(all_tours)
    
    return df_all_tours
```
<br>

### Length (duration/distance) of each tour


```python
def calc_tour_length(df_all_tours, df_routes):
    """
    Function calculates the total length of all possible tours
    Input:
        df_all_tours - df with all tours, output of get_all_possible_tours
        df_routes - df with all intermediate routes including length of route
    """
    
    # initialize empty numpy matrix
    dist_matrix = np.empty(df_all_tours.shape)
    
    # fill matrix with duration/distance
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]-1):
            start = df_all_tours[j][i]
            end   = df_all_tours[j+1][i]
            mask = (df_routes["location_x"] == start) & (df_routes["location_y"] == end)
            dist_matrix[i][j] = df_routes[mask]["length"]
            
    total_length = dist_matrix.sum(axis=1)
    
    return total_length
```
<br>

## Main function


```python
def route_optimization(path_csv,
                       sep_csv,
                       do_round,
                       google_cloud_api_key,
                       mode_transport,
                       mode_metric,
                       print_stats):
    """
    Function calculates the best/worst route of a given set of locations.
    Input:
        path_csv - string, path to csv
        sep_csv - string, seperator used in csv
        do_round - boolean, if True -> starting point == end point
        google_cloud_api_key - string, personal google cloud api key
        mode_transport - string, either "walking", "driving" or "bicycling"
        mode_metric - string, either "duration" or "distance"
        print_stats - boolean, if True -> prints best/worst route
    """

    # read the csv file
    df_locations = read_locations_csv(path_csv, 
                                      sep_csv)

    # all intermediate routes
    df_routes = get_intermediate_routes(df_locations, 
                                        do_round)

    # list of google maps distance matrix results
    list_metric_result = query_google_maps_distance_matrix(df_routes, 
                                                           google_cloud_api_key, 
                                                           mode_transport, 
                                                           mode_metric)

    # join length to intermediate routes
    df_routes["length"] = list_metric_result

    # df with all possible tours
    df_all_tours = get_all_possible_tours(df_locations, 
                                          do_round)

    # calculate total length of all tours
    length_all_tours = calc_tour_length(df_all_tours, 
                                        df_routes)
    
    # final dataframe: all possible tours with corresponding length
    df_final = df_all_tours.copy()
    df_final["total_length"] = length_all_tours
    
    # print stats and optimal/worst route
    if print_stats == True:
        min_metric = df_final["total_length"].min()
        max_metric = df_final["total_length"].max()
        if mode_metric == "duration":
            print(f"Fastest time: {round(min_metric,2)} minutes.")
            print(f"Slowest time: {round(max_metric,2)} minutes.")
            print()
            print("Fastest route:")
            print(df_final[df_final["total_length"] == min_metric].iloc[0])
            print()
            print("Slowest route:")
            print(df_final[df_final["total_length"] == max_metric].iloc[0])
        else:
            print(f"Shortest distance: {round(min_metric,2)} meters.")
            print(f"Longest distance: {round(max_metric,2)} meters.")
            print()
            print("Shortest route:")
            print(df_final[df_final["total_length"] == min_metric].iloc[0])
            print()
            print("Longest route:")
            print(df_final[df_final["total_length"] == max_metric].iloc[0])

    return df_final
```

```python
# import packages
import os
import numpy as np
import pandas as pd
import googlemaps
from itertools import permutations

# request user input:
while True:
    to_optimize_nr = input("Was soll optimiert werden? Zeit (1) oder Distanz (2)? Eingabe: ")
    try:
        if (int(to_optimize_nr) < 1) | (int(to_optimize_nr) > 2):
            print("Das ist keine gültige Nummer! Bitte erneut eingeben.\n")
        else:
            to_optimize_nr = int(to_optimize_nr)
            if to_optimize_nr == 1:
                to_optimize_char = "duration"
            else:
                to_optimize_char = "distance"
            print("Es wird folgendes optimiert: " + to_optimize_char)
            print()
            print("Bitte warten ...")
            print()
            break
    except ValueError:
        print("Das ist keine gültige Nummer! Bitte erneut eingeben.\n")

# change working directory
os.chdir('c:/users/schmi/desktop/martin/python/google_maps_trips')

# google cloud api key for goole maps (routes)
google_cloud_api_key = "AIzaSyCg2K3TyP-4FDQm6hqQADDqxyK1P0BvVd0"

# locations (given in longitude and latitude)
orte_orig = pd.read_csv("theresienfeld.csv", sep = ";")
n_orte = orte_orig.shape[0]

# goolge maps api connection
gmaps = googlemaps.Client(key = google_cloud_api_key)

# cartesian product of all locations
# exclude routes from "a" to "a" and from "a" to "starting point"
orte = orte_orig.copy()
orte["join"] = 1
orte = pd.merge(orte, orte, how = "outer", on = "join")
filter = (orte["ort_x"] != orte["ort_y"])
orte = orte[filter]
del orte["join"]
del orte["start_y"]
orte.reset_index(inplace = True, drop = True)
n_pairs = orte.shape[0]

# start with empty list where we will store the walking distances in minutes
result = []

# query google maps routes api with different starting and end points
for entry in range(n_pairs):
    start = (orte["latitude_x"][entry],orte["longitude_x"][entry])
    end   = (orte["latitude_y"][entry],orte["longitude_y"][entry])
    tmp = gmaps.distance_matrix(start, end, mode='walking')["rows"][0]["elements"][0][to_optimize_char]["value"]
    if to_optimize_char == "duration":
        tmp = tmp/60
    else:
        tmp = tmp
    result.append(tmp)
    
# add the walking distances in minutes to the original table
orte["wert"] = result

# create all possible combinations of doing a tour
visiting_locations = orte_orig[orte_orig["start"] == 0]["ort"].tolist()
starting_location  = orte_orig[orte_orig["start"] == 1]["ort"].tolist()[0]
shuffle_visiting_locations = list(permutations(visiting_locations, n_orte -1))

# create a list of all possible tours
all_tours = []
for tmp_tour in shuffle_visiting_locations:
    all_tours.append((starting_location,) + tmp_tour + (starting_location,))
    #all_tours.append((starting_location,) + tmp_tour)
all_tours = pd.DataFrame(all_tours)
n_tours = all_tours.shape[0]


dist_matrix = np.empty((n_tours,n_orte))

for i in range(n_tours):
    for j in range(n_orte):
        start = all_tours[j][i]
        end   = all_tours[j+1][i]
        filter = (orte["ort_x"] == start) & (orte["ort_y"] == end)
        dist_matrix[i][j] = orte[filter]["wert"]
        
all_tours["total"] = dist_matrix.sum(axis = 1)

min_time = all_tours["total"].min()

filter = (all_tours["total"] == min_time)

final = all_tours[filter]
final.reset_index(inplace = True, drop = True)

max_range = final.shape[1]
for i in range(max_range-1):
    print(i+1,  "--->", final[i][0])
    
if to_optimize_char == "distance":
    einheit = "meter"
else:
    einheit = "minuten"

print()
print("Total time/distance: ", final["total"][0], einheit)
print()
print("Anzahl verschiedener getesteter Routen: ", all_tours.shape[0])
print()
print("Anzahl Anfragen an Google Maps Distance Api: ", n_pairs)
```

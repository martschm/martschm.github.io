Import required packages

```python
import os
import numpy as np
import pandas as pd
import googlemaps
from itertools import permutations
```

Request user input. The user can choose whether she/he want's to optimize time or distance

```python
while True:
    to_optimize_nr = input("What do you want to optimize? Time (1) or Distance (2)? Input: ")
    try:
        if (int(to_optimize_nr) < 1) | (int(to_optimize_nr) > 2):
            print("That's not a valid number! Please enter again.\n")
        else:
            to_optimize_nr = int(to_optimize_nr)
            if to_optimize_nr == 1:
                to_optimize_char = "duration"
            else:
                to_optimize_char = "distance"
            print("The following will be optimized: " + to_optimize_char)
            print()
            print("Please wait ...")
            print()
            break
    except ValueError:
        print("That's not a valid number! Please enter again.\n")
```

Change working directory to where the csv-file with locations is located
```python
os.chdir('your_working_directory_here')
```

Google Maps Distance Matrix API connection
```python
google_cloud_api_key = "your_api_key"
gmaps = googlemaps.Client(key = google_cloud_api_key)
```

Read in the data: locations (given in longitude and latitude)
```python
data = pd.read_csv("locations.csv", sep = ";")
n_places = data.shape[0]
```

Here is an example of how the csv-file should be structured<br><br>
<img src="images/python_route_optimization_sample_csv.png?raw=true"/>


Exclude routes from "A" to "A" and from "A" to "starting point"
```python
locations = data.copy()
locations["join"] = 1
locations = pd.merge(locations, locations, how = "outer", on = "join")
filter = (locations["ort_x"] != locations["ort_y"])
locations = locations[filter]
del locations["join"]
del locations["start_y"]
locations.reset_index(inplace = True, drop = True)
n_pairs = locations.shape[0]
```

I start with an empty list where I will store the walking distances in minutes.<br>
I query google maps routes api with different starting and end points
```python
result = []
for entry in range(n_pairs):
    start = (locations["latitude_x"][entry],locations["longitude_x"][entry])
    end   = (locations["latitude_y"][entry],locations["longitude_y"][entry])
    tmp = gmaps.distance_matrix(start, end, mode='walking')["rows"][0]["elements"][0][to_optimize_char]["value"]
    if to_optimize_char == "duration":
        tmp = tmp/60
    else:
        tmp = tmp
    result.append(tmp)
```
    
I add the walking distances in minutes to the original table and create all possible combinations of doing a tour
```python
locations["value"] = result

visiting_locations = data[data["start"] == 0]["ort"].tolist()
starting_location  = data[data["start"] == 1]["ort"].tolist()[0]
shuffle_visiting_locations = list(permutations(visiting_locations, n_places -1))

all_tours = []
for tmp_tour in shuffle_visiting_locations:
    all_tours.append((starting_location,) + tmp_tour + (starting_location,))
all_tours = pd.DataFrame(all_tours)
n_tours = all_tours.shape[0]
```
Create a matrix with time/distances of all possible tours und select the tour which minimizes time/distance
```python
dist_matrix = np.empty((n_tours,n_places))

for i in range(n_tours):
    for j in range(n_places):
        start = all_tours[j][i]
        end   = all_tours[j+1][i]
        filter = (locations["ort_x"] == start) & (locations["ort_y"] == end)
        dist_matrix[i][j] = locations[filter]["value"]
        
all_tours["total"] = dist_matrix.sum(axis = 1)

min_time = all_tours["total"].min()

filter = (all_tours["total"] == min_time)

final = all_tours[filter]
final.reset_index(inplace = True, drop = True)
```
Print the best route to the console
```python
max_range = final.shape[1]
for i in range(max_range-1):
    print(i+1,  "--->", final[i][0])
```
Print final stats
```python
if to_optimize_char == "distance":
    unit = "meter"
else:
    unit = "minutes"

print("Total time/distance: ", final["total"][0], einheit)
print("Number of different routes tested: ", all_tours.shape[0])
print("Number of requests to Google Maps Distance API: ", n_pairs)
```

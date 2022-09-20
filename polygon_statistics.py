##
import time
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from sentinelhub import (
    CRS,
    Geometry,
    BBox,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    SHConfig,
    bbox_to_dimensions
)
import matplotlib.pyplot as plt

##
gdf = gpd.read_file("./python-task/land_polygons.shp")

# map land type to index
land_type_map = {}
for idx, land_type in enumerate(gdf.LAND_TYPE.unique()):
    land_type_map[land_type] = idx

# obtain lower left and upper right coordinates of whole area
bbox_list = [Geometry(i, crs=CRS.UTM_33N).bbox for i in gdf.geometry]
min_x = np.min([i.min_x for i in bbox_list])
min_y = np.min([i.min_y for i in bbox_list])
max_x = np.max([i.max_x for i in bbox_list])
max_y = np.max([i.max_y for i in bbox_list])
aggregate_bbox = BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.UTM_33N)
aggregate_bbox_size = bbox_to_dimensions(aggregate_bbox, resolution=10)
##
# visualize all polygons
gdf.plot()
plt.axis('off')
plt.show()
##
config = SHConfig()
evalscript = """
    // VERSION=3
    
    function setup() {
        return {
            input: [
                {
                    bands: ["B04", "B08"],
                    units: ["REFLECTANCE", "REFLECTANCE"]
                }
            ],
            output: {
                bands: 1,
                sampleType:"FLOAT32"
            }
        }
    }
    function evaluatePixel(sample) {    
        let ndvi = index(sample.B08, sample.B04);
        return [ndvi];
    }
"""

request_ndvi = SentinelHubRequest(evalscript=evalscript,
                                  input_data=[
                                      SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C,
                                                                    time_interval=("2018-09-28", "2018-09-29"), )
                                  ],
                                  responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                                  bbox=aggregate_bbox,
                                  resolution=(10, 10),
                                  config=config,
                                  data_folder="./ndvi_data/"
                                  )
ndvi_image = request_ndvi.get_data()[0]  # shape: H * W
# request_ndvi.save_data()
##
# read data in TIFF format
dataset = rasterio.open("./ndvi_data/1508bf73a45b55a6e1755de1527c7c56/response.tiff")
ndvi_value = dataset.read(1)  # shape: H * W

# obtain geojson data, polygon index and land type of different polygons
shapes = [(Geometry(gdf.geometry.iloc[i], crs=CRS.UTM_33N).get_geojson(), i, land_type_map[gdf.LAND_TYPE.iloc[i]]) for
          i in range(gdf.shape[0])]

# rasterize polygon index and land type to map
polygon_image = features.rasterize(((i[0], i[1]) for i in shapes), out_shape=dataset.shape, fill=-1,
                                   transform=dataset.transform)  # shape: H * W
land_type_image = features.rasterize(((i[0], i[2]) for i in shapes), out_shape=dataset.shape, fill=-1,
                                     transform=dataset.transform)  # shape: H * W


##
def get_statistics(query):
    if isinstance(query, str):

        # index specific land type or polygon on map
        index = land_type_image == land_type_map[query]  # bool: H * W

        # obtain ndvi values with same index
        array = ndvi_value[index]  # 1-d array. length: number of pixels
        return (array.min().round(6), array.max().round(6), array.mean().round(6), array.std().round(6))
    elif isinstance(query, (int, np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64)):
        index = polygon_image == query
        array = ndvi_value[index]
        return (array.min().round(6), array.max().round(6), array.mean().round(6), array.std().round(6))
    else:
        raise TypeError("Input must be a string or integer")


##
def shortest_distance(array):
    # time complexity 0(NlogN)
    sorted_array = sorted(array)
    sorted_id = sorted(range(len(array)), key=lambda x: array[x])
    resValue = sorted_array[1] - sorted_array[0]
    resIndex = (0, 1)
    idx_list = []
    for i in range(len(array) - 1):
        currentValue = sorted_array[i + 1] - sorted_array[i]
        if currentValue <= resValue:
            resValue = currentValue
            resIndex = (i, i + 1)
            idx_list.append((sorted_id[resIndex[0]], sorted_id[resIndex[1]]))

    idx_list = [i for i in idx_list if array[i[1]] - array[i[0]] == min([array[i[1]] - array[i[0]] for i in idx_list])]
    idx_list = [(i[1], i[0]) if i[0] > i[1] else i for i in idx_list]
    return idx_list


##
# calculate pair with the shortest distance in a number of points
def distance(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def bruteForce(P, n):
    min_val = float('inf')
    pair = (P[0], P[1])
    for i in range(n):
        for j in range(i + 1, n):
            if distance(P[i], P[j]) < min_val:
                min_val = distance(P[i], P[j])
                pair = (P[i], P[j])
    return min_val, pair


def stripClosest(strip, size, d):
    pair = None
    if size <= 1:
        return d, pair
    else:
        min_val = d
        for i in range(size):
            j = i + 1
            while j < size and (strip[j][1] - strip[i][1]) < min_val:
                if distance(strip[i], strip[j]) < min_val:
                    min_val = distance(strip[i], strip[j])
                    pair = (strip[i], strip[j])
                j += 1
        return min_val, pair


def closestUtil(P, n):
    if n <= 3:
        return bruteForce(P, n)
    mid = n // 2
    midPoint = P[n // 2]
    Pl = P[:mid]
    Pr = P[mid:]
    dl, pairL = closestUtil(Pl, mid)
    dr, pairR = closestUtil(Pr, n - mid)
    d = min(dl, dr)
    if d == dl:
        pair = pairL
    if d == dr:
        pair = pairR
    stripP = []
    lr = Pl + Pr
    for i in range(n):
        if abs(lr[i][0] - midPoint[0]) < d:
            stripP.append(lr[i])
    stripP.sort(key=lambda x: x[1])
    dstrip, pairS = stripClosest(stripP, len(stripP), d)
    min_val = min(d, dstrip)
    # print(dl, pairL, dr, pairR, dstrip, pairS)
    if pairS is not None and min_val == dstrip:
        pair = pairS
        return min_val, pair
    else:
        return min_val, pair


def closest(P, n):
    # time complexity O(NlogN)
    P_sorted = sorted(P, key=lambda x: x[0])
    return closestUtil(P_sorted, n)


##
def get_closest_pair(id_list, *criteria):
    num_id, num_criteria = len(id_list), len(criteria)

    def get_method(array, method):
        if method == 'min':
            return array.min()
        elif method == 'max':
            return array.max()
        elif method == 'mean':
            return array.mean()
        elif method == 'std':
            return array.std()

    # flatten polygon map and ndvi value map to get valid indexes more easily
    # remove pixels that don't belong to any polygons to save calculation time
    polygon_index = np.argwhere(polygon_image.flatten() != -1)
    ndvi_array = ndvi_value.flatten()[polygon_index]
    polygon_index_array = polygon_image.flatten()[polygon_index]
    matrix = np.zeros((num_id, num_criteria))  # shape: num_id * num_criteria
    for i in range(num_id):
        array = ndvi_array[polygon_index_array == id_list[i]]
        # save statistics in matrix
        matrix[i][0] = get_method(array, criteria[0])
        if num_criteria > 1:
            matrix[i][1] = get_method(array, criteria[1])

    # calculate absolute distance
    if num_criteria == 1:
        sorted_distance = shortest_distance(matrix.reshape(-1))
        pair = [(id_list[i[0]], id_list[i[1]]) for i in sorted_distance]
    elif num_criteria == 2:
        pairs = list(zip(matrix[:, 0], matrix[:, 1]))
        distance, min_pair = closest(pairs, len(matrix))
        idx1, idx2 = pairs.index(min_pair[0]), pairs.index(min_pair[1])
        pair = [(id_list[idx1], id_list[idx2])]

    if len(pair) == 1:
        return pair[0]
    else:
        return pair


time_start = time.time()
print(get_closest_pair([2, 3, 5, 7, 11], 'min'))
print(get_closest_pair(list(range(10000)), 'mean', 'std'))
time_end = time.time()
time_sum = time_end - time_start
print(time_sum)

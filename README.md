This task seeks to obtain NDVI value (normalized difference vegetation index) of a series of rasterized polygons
in a specific area with SentinelHub, an API from Sentinul-2 LLC Satellite. The pixel-wise numeric value represents
the daily NDVI value of a 10m*10m area. Then an algorithm is designed to find two polygons that shares the closest
difference in one or two of these four statistics: minimum NDVI value of the polygon, maximum, mean and standard deviation.

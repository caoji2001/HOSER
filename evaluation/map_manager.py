import math
import geopy
import geopy.distance

class MapManager(object):
    def __init__(self, city):
        assert city in ['BJ_Taxi', 'Porto_Taxi', 'SF_Taxi']

        if city == 'BJ_Taxi':
            self.lon_0 = 116.25
            self.lon_1 = 116.25+0.26
            self.lat_0 = 39.79
            self.lat_1 = 39.79+0.21
        elif city == 'Porto_Taxi':
            self.lon_0 = -8.6890
            self.lon_1 = -8.6890+0.1313
            self.lat_0 = 41.1406
            self.lat_1 = 41.1406+0.0449
        else:
            self.lon_0 = -122.5123
            self.lon_1 = -122.3617
            self.lat_0 = 37.7083
            self.lat_1 = 37.8309

        self.img_width = math.ceil(geopy.distance.geodesic((self.lat_0, self.lon_0), (self.lat_0, self.lon_1)).meters / 200) + 1
        self.img_height = math.ceil(geopy.distance.geodesic((self.lat_0, self.lon_0), (self.lat_1, self.lon_0)).meters / 200) + 1

    def gps2grid(self, lon, lat):
        x = math.floor(geopy.distance.geodesic((lat, self.lon_0), (lat, lon)).meters / 200)
        y = math.floor(geopy.distance.geodesic((self.lat_0, lon), (lat, lon)).meters / 200)
        assert 0 <= x < self.img_width
        assert 0 <= y < self.img_height
        return x, y

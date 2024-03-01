# # -*- coding: utf-8 -*-

import unittest
from pyedautils.meteo_swiss import get_current_station_data, find_nearest_station
from pyedautils.geopy import get_altitude_lat_long, get_coordindates_ch_plz

class TestGeopyhelpers(unittest.TestCase):
    """Tests for "geopyhelpers.py"."""
    
    def test_get_current_station_data(self):
        """Test get_current_station_data()"""
        self.assertGreaterEqual(get_current_station_data().shape[0], 1)
        
    def test_find_nearest_station(self):
        coord = get_coordindates_ch_plz(6048)
        altitude = get_altitude_lat_long(coord[0], coord[1])
        self.assertEqual(find_nearest_station(coord[0], coord[1], altitude, sensor="temp"),"LUZ")
        coord = get_coordindates_ch_plz(6197)
        altitude = get_altitude_lat_long(coord[0], coord[1])
        self.assertEqual(find_nearest_station(coord[0], coord[1], altitude, sensor="temp"), "FLU")
        self.assertEqual(find_nearest_station(coord[0], coord[1], altitude, sensor="relhum"), "FLU")
        self.assertEqual(find_nearest_station(coord[0], coord[1], altitude, sensor="globrad"), "BAN")
        self.assertEqual(find_nearest_station(coord[0], coord[1], altitude, sensor="rain"), "FLU")
        
if __name__ == '__main__':
    unittest.main()

# # -*- coding: utf-8 -*-

import unittest
from pyedautils.geopy import get_altitude_lat_long, get_lat_long_address, get_altitude_lv95, convert_wsg84_to_lv95, GeocodingError, get_coordindates_ch_plz, get_distance_between_two_points

class TestGeopyhelpers(unittest.TestCase):
    """Tests for "geopyhelpers.py"."""
    
    def test_get_altitude_lat_long(self):
        """Test get_altitude() with different lat/long pairs"""
        self.assertIn(round(get_altitude_lat_long(47.01450, 8.30620),0), range(438,445))
        with self.assertRaises(ValueError):
            get_altitude_lat_long(57.01450, 8.30620)
        with self.assertRaises(ValueError):
            get_altitude_lat_long(100, 10)
        with self.assertRaises(ValueError):
            get_altitude_lat_long(-100, 10)
        with self.assertRaises(ValueError):
            get_altitude_lat_long(10, -200)
        with self.assertRaises(ValueError):
            get_altitude_lat_long(10, 200)

    def test_get_lat_long_address(self):
        """Test get_lat_long() with different addresses"""
        self.assertEqual(round(get_lat_long_address("Technikumstrasse 21, 6048 Horw")[0], 1), 47.0)
        self.assertEqual(round(get_lat_long_address("Technikumstrasse 21, 6048 Horw")[1], 1), 8.3)
        with self.assertRaises(GeocodingError):
            get_lat_long_address("Highway to hell, Horw, Switzerland")        
        
    def test_convert_wsg84_to_lv95(self):
        """Test convert_wsg84_to_lv95() with different coodrinates"""
        self.assertIn(round(convert_wsg84_to_lv95(47.01331, 8.30612)[0], 0), range(2665959,2665962))
        self.assertIn(round(convert_wsg84_to_lv95(47.01450, 8.30620)[1], 0), range(1207300,1207450))
        with self.assertRaises(GeocodingError):
            convert_wsg84_to_lv95(57.01450, 8.30620)
        with self.assertRaises(GeocodingError):
            convert_wsg84_to_lv95(47.01450, 15.30620)
    
    def test_get_altitude_lv95(self):
        """Test get_altitude_ch_lv95() with different lat/long pairs"""
        self.assertIn(round(get_altitude_lv95([2665949.0,1207341.8]),0), range(438,445))
        self.assertIn(round(get_altitude_lv95(convert_wsg84_to_lv95(47.01450, 8.30620)),0), range(438,445))
        
        
    def test_find_coordindates_ch_plz(self):
        self.assertAlmostEqual(get_coordindates_ch_plz(6048)[0], 47.01, delta=0.1)
        self.assertAlmostEqual(get_coordindates_ch_plz(6048)[1], 8.3, delta=0.1)
        with self.assertRaises(GeocodingError):
            get_coordindates_ch_plz(424242)
                        
if __name__ == '__main__':
    unittest.main()

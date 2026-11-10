import unittest
from xal_temporal.geometry import rescue_painted_extent
class PaintedFusionTests(unittest.TestCase):
 def test_recover_legs_preserve_both_horizontal_extents(self):
  coarse=[877,739,1038,771];local=[877.3,725.3,1040.5,844.1]
  self.assertEqual(rescue_painted_extent(coarse,local,(2048,1152),.58,True),[877.,725.3,1040.5,844.1])
 def test_roof_tree_expansion_rejected(self):
  self.assertIsNone(rescue_painted_extent([835,787,1031,829],[835.4,663.3,1227,911.4],(2048,1152),.58,True))
  self.assertIsNone(rescue_painted_extent([912,712,1080,738],[912.3,640.3,1214.5,804.7],(2048,1152),.58,True))
  self.assertIsNone(rescue_painted_extent([809,666,958,695],[815.4,631.6,1017.1,741.6],(2048,1152),.58,True))
 def test_no_neighbour_or_missing_detection_is_not_synthesized(self):
  self.assertIsNone(rescue_painted_extent([877,739,1038,771],[877.3,725.3,1040.5,844.1],(2048,1152),.58,False))
  self.assertIsNone(rescue_painted_extent(None,[877,725,1040,844],(2048,1152),.58,True))
if __name__=='__main__':unittest.main()

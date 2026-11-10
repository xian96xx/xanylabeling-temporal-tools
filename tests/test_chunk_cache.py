import os, tempfile, unittest
from pathlib import Path
import numpy as np
from xal_temporal.cache import TrackingChunkCache

class ChunkCacheTests(unittest.TestCase):
 def test_keys_cover_content_order_seed_mode_and_margins(self):
  with tempfile.TemporaryDirectory() as tmp:
   a,b=Path(tmp)/'a',Path(tmp)/'b';a.write_bytes(b'abc');b.write_bytes(b'def');c=TrackingChunkCache();fp=c.fingerprints([a,b]);base=c.key(fp,[1,2,3,4],None,False,None)
   self.assertNotEqual(base,c.key(c.fingerprints([b,a]),[1,2,3,4],None,False,None))
   self.assertNotEqual(base,c.key(fp,[1,2,3,5],None,False,None))
   self.assertNotEqual(base,c.key(fp,[1,2,3,4],None,True,{'pixels':[1]*4}))
   old=a.stat();a.write_bytes(b'xyz');os.utime(a,ns=(old.st_atime_ns,old.st_mtime_ns));self.assertNotEqual(base,c.key(c.fingerprints([a,b]),[1,2,3,4],None,False,None))
   mask=np.zeros((3,5),bool);k=c.key(fp,None,mask,False,None);mask[0,0]=True;self.assertNotEqual(k,c.key(fp,None,mask,False,None))
 def test_values_are_independent_and_eviction_is_bounded(self):
  c=TrackingChunkCache(limit=1);mask=np.array([[True,False,True],[False,True,False]]);c.put('x',[[1.,2.,3.,4.]],[[2.,3.,4.,5.]],mask);mask[:]=False
  boxes,evidence,carry=c.get('x');self.assertTrue(carry[0,0]);boxes[0][0]=999;self.assertEqual(c.get('x')[0][0][0],1.);c.put('y',[None],[None],np.zeros((2,3),bool));self.assertIsNone(c.get('x'));self.assertIsNotNone(c.get('y'))
if __name__=='__main__':unittest.main()

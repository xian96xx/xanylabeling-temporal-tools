import io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from xal_temporal import tracking as t
from xal_temporal import masks

class MaskPromptTests(unittest.TestCase):
 def test_roundtrip_binding_and_integrity(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); image=root/'image.jpg';Image.new('RGB',(64,48)).save(image);st=image.stat();rows=[{'image':image,'path':image.with_suffix('.json'),'size':(64,48),'image_stamp':[st.st_size,st.st_mtime_ns]}];anchor={0:[10,10,40,30]};points={0:[[12,12,1]]};im=Image.new('RGBA',(64,48),(0,0,0,0));im.putpixel((12,12),(25,235,130,115));b=io.BytesIO();im.save(b,format='PNG')
   with patch.object(t,'PROJECT_ROOT',root),patch.object(masks,'PROJECT_ROOT',root):
    rec=masks.save_mask(rows,('drone',1),0,anchor[0],points[0],b.getvalue());self.assertEqual(masks.load_masks(rows,('drone',1),anchor,points),{0:rec});self.assertEqual(masks.load_masks(rows,('drone',1),anchor,{0:[[13,12,1]]}),{});self.assertEqual(masks.load_masks(rows,('drone',1),{0:[11,10,40,30]},points),{})
    Path(rec['path']).write_bytes(b'altered');self.assertEqual(masks.load_masks(rows,('drone',1),anchor,points),{});self.assertFalse(rows[0]['path'].exists())
if __name__=='__main__':unittest.main()

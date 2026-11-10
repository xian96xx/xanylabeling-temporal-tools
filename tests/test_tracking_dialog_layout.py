import os,tempfile,unittest,json
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PIL import Image
from PyQt6 import QtCore,QtGui,QtWidgets
from xal_temporal.qt.tracking_dialog import TemporalTrackingDialog

class TrackingLayoutTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_small_window_keeps_actions_visible_and_controls_scroll(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);ims=[]
   for i in range(3):
    image=p/f'{i:04}.jpg';Image.new('RGB',(100,80)).save(image);ims.append(image)
    image.with_suffix('.json').write_text(json.dumps({'imageWidth':100,'imageHeight':80,'shapes':[{'label':'drone','group_id':1,'shape_type':'rectangle','points':[[10,10],[30,30]]}]}),encoding='utf8')
   d=TemporalTrackingDialog(None,ims,[x.with_suffix('.json') for x in ims],[('drone',1)],0,('drone',1));d.show();d.resize(1150,600);self.app.processEvents()
   pos=d.apply.mapTo(d,QtCore.QPoint(0,0));self.assertLessEqual(pos.y()+d.apply.height(),d.height());self.assertLessEqual(d.minimumSizeHint().height(),600)
   bar=d.controls_scroll.verticalScrollBar();self.assertGreater(bar.maximum(),0);bar.setValue(bar.maximum());self.app.processEvents();self.assertEqual(bar.value(),bar.maximum());d.close()
if __name__=='__main__':unittest.main()

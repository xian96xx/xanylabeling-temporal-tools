import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xal_temporal import smoothing as m


class SmoothingTests(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array([[20+i+(i%2)*.7, 20, 40+i+(i%2)*.7, 40+(i%3)*.5] for i in range(15)],float)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images=[];self.paths=[]
        for i,b in enumerate(self.boxes):
            image=self.root/f'{i+1:05d}.jpg';image.write_bytes(b'image')
            p=image.with_suffix('.json')
            doc={'imageWidth':200,'imageHeight':100,'custom':{'preserve':True},'shapes':[
                {'label':'truck','group_id':2,'shape_type':'rectangle','points':[b[:2].tolist(),b[2:].tolist()],'flags':{'keep':'yes'}},
                {'label':'other','group_id':4,'shape_type':'rectangle','points':[[1,1],[10,10]]}]}
            p.write_text(json.dumps(doc),encoding='utf-8');self.images.append(image);self.paths.append(p)

    def tearDown(self):self.temp.cleanup()

    def preview(self, **kwargs):
        return m.build_preview(self.images,self.paths,('truck',2),{6},**kwargs)

    def test_anchors_limit_and_roughness(self):
        out=m.smooth_boxes(self.boxes,{6},limit=.5)
        np.testing.assert_array_equal(out[[0,6,-1]],self.boxes[[0,6,-1]])
        self.assertLessEqual(abs(out-self.boxes).max(),.500001)
        self.assertLess(np.linalg.norm(np.diff(out,n=2,axis=0)),np.linalg.norm(np.diff(self.boxes,n=2,axis=0)))

    def test_modes_and_interpolation(self):
        for method in ('smooth','interpolate'):
            pos=m.smooth_boxes(self.boxes,{6},mode='position',method=method)
            size=m.smooth_boxes(self.boxes,{6},mode='size',method=method)
            np.testing.assert_allclose(pos[:,2:]-pos[:,:2],self.boxes[:,2:]-self.boxes[:,:2])
            np.testing.assert_allclose(size[:,2:]+size[:,:2],self.boxes[:,2:]+self.boxes[:,:2])

    def test_small_boxes_and_image_edges(self):
        b=np.array([[0,0,1,1],[.9,.9,1.1,1.1],[0,0,1,1],[.9,.9,1.1,1.1]],float)
        for mode in ('both','position','size'):
            out=m.smooth_boxes(b,set(),limit=50,mode=mode,image_size=(2,2))
            self.assertTrue(np.all(out[:,:2]>=0));self.assertTrue(np.all(out[:,2:]<=2));self.assertTrue(np.all(out[:,2:]>out[:,:2]))

    def test_roundtrip_and_metadata(self):
        original=[p.read_bytes() for p in self.paths]
        rows=self.preview()
        for r in rows:
            before=copy.deepcopy(r['doc']);after=json.loads(r['after'])
            before['shapes'][0]['points']=after['shapes'][0]['points']
            self.assertEqual(before,after)
        backup=m.apply_preview(rows)
        self.assertEqual(m.latest_backup(self.root),backup)
        m.undo_preview(backup,self.root)
        self.assertEqual(original,[p.read_bytes() for p in self.paths])
        self.assertIsNone(m.latest_backup(self.root))

    def test_preview_conflict(self):
        rows=self.preview();self.paths[2].write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'重新生成'):m.apply_preview(rows)
        self.assertFalse((self.root/'.temporal_smoothing_backups').exists())

    def test_undo_conflict_changes_nothing(self):
        backup=m.apply_preview(self.preview());self.paths[3].write_text('{}',encoding='utf-8')
        snapshot=[p.read_bytes() for p in self.paths]
        with self.assertRaisesRegex(ValueError,'撤销已停止'):m.undo_preview(backup,self.root)
        self.assertEqual(snapshot,[p.read_bytes() for p in self.paths])

    def test_failure_rolls_back(self):
        original=[p.read_bytes() for p in self.paths];actual=m.atomic_write
        def fail(path,raw):
            if Path(path)==self.paths[3] and raw!=original[3]:raise OSError('simulated disk failure')
            actual(path,raw)
        with patch.object(m,'atomic_write',side_effect=fail):
            with self.assertRaises(OSError):m.apply_preview(self.preview())
        self.assertEqual(original,[p.read_bytes() for p in self.paths])

    def test_missing_duplicate_and_seam_rejected(self):
        p=self.paths[3];original=p.read_bytes();d=json.loads(original)
        d['shapes'].append(copy.deepcopy(d['shapes'][0]));p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'多个框'):self.preview()
        p.write_bytes(original);d=json.loads(original);d['shapes'][0]['points']=[[180,20],[199,40]];p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'接缝'):self.preview()
        p.write_text(json.dumps({'imageWidth':200,'imageHeight':100,'shapes':[]}))
        with self.assertRaisesRegex(ValueError,'缺失'):self.preview()

    def test_four_point_order(self):
        shape={'points':[[40,40],[20,40],[20,20],[40,20]]}
        m.replace_bounds(shape,[21,22,43,44])
        self.assertEqual(shape['points'],[[43,44],[21,44],[21,22],[43,22]])

    def test_manual_edge_priority_and_local_recompute(self):
        base,_,_,_=m.refine_boxes(self.boxes,{4,10},(200,100))
        edited,working,fixed,manual=m.refine_boxes(
            self.boxes,{4,10},(200,100),manual_overrides={7:{0:self.boxes[7,0]+3}}
        )
        self.assertEqual(edited[7,0],self.boxes[7,0]+3)
        self.assertTrue(fixed[7,0] and manual[7,0])
        np.testing.assert_array_equal(edited[:5,0],base[:5,0])
        np.testing.assert_array_equal(edited[10:,0],base[10:,0])
        np.testing.assert_array_equal(edited[:,1:],base[:,1:])
        self.assertLessEqual(abs(edited-working).max(),1.000001)

    def test_disabled_edges_and_per_edge_lock(self):
        out,_,fixed,_=m.refine_boxes(self.boxes,set(),(200,100),enabled_edges=[False,True,False,True],edge_locks={7:{1}})
        np.testing.assert_array_equal(out[:,[0,2]],self.boxes[:,[0,2]])
        self.assertEqual(out[7,1],self.boxes[7,1]);self.assertTrue(fixed[7,1])

    def test_manual_apply_undo_and_invalid_correction(self):
        original=[p.read_bytes() for p in self.paths]
        rows=self.preview(manual_overrides={7:{0:self.boxes[7,0]+3}})
        backup=m.apply_preview(rows)
        actual=json.loads(self.paths[7].read_text())['shapes'][0]['points'][0][0]
        self.assertEqual(actual,self.boxes[7,0]+3)
        m.undo_preview(backup,self.root)
        self.assertEqual(original,[p.read_bytes() for p in self.paths])
        with self.assertRaisesRegex(ValueError,'人工修正'):
            self.preview(manual_overrides={7:{0:500}})

    def test_review_list_and_coupled_locks(self):
        rows=self.preview(limit=.1)
        flags=m.review_flags(rows,.1)
        self.assertTrue(flags)
        out,working,fixed,_=m.refine_boxes(self.boxes,set(),(200,100),mode='size',edge_locks={7:{0}})
        self.assertTrue(fixed[7,0] and fixed[7,2])
        np.testing.assert_allclose(out[:,2:]+out[:,:2],working[:,2:]+working[:,:2])


if __name__=='__main__':unittest.main(verbosity=2)

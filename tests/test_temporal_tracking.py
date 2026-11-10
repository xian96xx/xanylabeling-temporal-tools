import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np
from xal_temporal.geometry import (mask_box, learn_edge_margins,
    apply_edge_margins, preserve_nearby_outer_edges, local_crop_window,
    crop_box_coordinates, combine_local_extent, interval_reference_aspect, local_size_supported)

from xal_temporal import tracking as m


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images, self.paths = [], []
        self.identity = ('car', 1)
        for i in range(7):
            image = self.root/f'{i+1:05d}.jpg'
            Image.new('RGB', (100, 80)).save(image)
            path = image.with_suffix('.json')
            doc = {'imageWidth': 100, 'imageHeight': 80, 'imagePath': image.name,
                   'imageData': None, 'custom': {'keep': True}, 'shapes': [
                       {'label': 'car', 'group_id': 1, 'shape_type': 'rectangle',
                        'points': [[10, 10], [30, 10], [30, 30], [10, 30]],
                        'attributes': {'target_id': 'target_1', 'mask': 'old mask'}, 'flags': {}, 'score': .9},
                       {'label': 'other', 'group_id': 2, 'shape_type': 'rectangle', 'points': [[1, 1], [5, 5]]}]}
            if i in (0, 6): path.write_text(json.dumps(doc), encoding='utf-8')
            self.images.append(image); self.paths.append(path)
        self.rows = m.load_sequence(self.images, self.paths, self.identity)
        self.anchors = {0: [10, 10, 30, 30], 6: [10, 10, 30, 30]}
        self.predictions = {i: {'box': [10+i, 10, 30+i, 30], 'reasons': []} for i in range(7)}

    def tearDown(self): self.temp.cleanup()

    def test_sparse_annotations_roundtrip_and_metadata(self):
        self.predictions[3]['box'] = None
        before = [m.read_optional(p) for p in self.paths]
        backup, count = m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        self.assertEqual(count, 4)
        self.assertFalse(self.paths[3].exists())
        self.assertEqual(self.paths[0].read_bytes(), before[0])
        created = json.loads(self.paths[1].read_text(encoding='utf-8'))['shapes'][0]
        self.assertEqual(created['attributes']['target_id'], 'target_1')
        self.assertNotIn('mask', created['attributes'])
        self.assertIsNone(created['score'])
        self.assertEqual(len(created['points']), 4)
        m.undo_tracking(backup, self.root)
        self.assertEqual(before, [m.read_optional(p) for p in self.paths])

    def test_existing_other_targets_and_fields_preserved(self):
        self.anchors[6] = [12, 12, 32, 32]
        m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        after = json.loads(self.paths[6].read_text(encoding='utf-8'))
        self.assertEqual(after['custom'], {'keep': True})
        self.assertEqual(after['shapes'][1], self.rows[6]['doc']['shapes'][1])
        self.assertEqual(after['shapes'][0]['points'][0], [12.0, 12.0])

    def test_assistant_initialization_is_not_marked_human(self):
        self.anchors[3] = [12., 12., 32., 32.]
        self.predictions[3]['prompt_source'] = 'assistant'
        m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        shape = json.loads(self.paths[3].read_text(encoding='utf-8'))['shapes'][0]
        self.assertTrue(shape['flags']['temporal_ai_generated'])
        self.assertFalse(shape['attributes']['temporal_ai_tracking']['human_keyframe'])
        self.assertEqual(shape['attributes']['temporal_ai_tracking']['prompt_source'], 'assistant')
        self.assertEqual(self.paths[0].read_bytes(), self.rows[0]['raw'])

    def test_new_file_created_by_other_program_blocks_entire_apply(self):
        self.paths[3].write_text('{}')
        before = [m.read_optional(p) for p in self.paths]
        with self.assertRaisesRegex(ValueError, '预览后被修改'):
            m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        self.assertEqual(before, [m.read_optional(p) for p in self.paths])

    def test_apply_failure_removes_new_files(self):
        before = [m.read_optional(p) for p in self.paths]
        real = m.atomic_write
        def fail(path, raw):
            if Path(path) == self.paths[4]: raise OSError('simulated disk error')
            real(path, raw)
        with patch.object(m, 'atomic_write', side_effect=fail):
            with self.assertRaises(OSError): m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        self.assertEqual(before, [m.read_optional(p) for p in self.paths])

    def test_undo_conflict_preserves_every_file(self):
        backup, _ = m.apply_draft(self.rows, self.predictions, self.anchors, self.identity)
        self.paths[2].write_text('{}')
        before = [m.read_optional(p) for p in self.paths]
        with self.assertRaisesRegex(ValueError, '撤销已停止'): m.undo_tracking(backup, self.root)
        self.assertEqual(before, [m.read_optional(p) for p in self.paths])

    def test_manual_frames_survive_review_and_only_nearby_intervals_change(self):
        anchors = {0: [10, 10, 30, 30], 3: [12, 12, 32, 32], 6: [10, 10, 30, 30]}
        output = m.review_predictions(self.predictions, anchors, self.rows)
        self.assertEqual(output[3]['box'], anchors[3])
        self.assertEqual(m.affected_span(anchors, 2, 0, 6), (0, 3))
        self.assertEqual(m.tracking_intervals(anchors, 0, 6), [(0, 3), (3, 6)])
        m.save_keyframes(self.rows, self.identity, anchors)
        restored, stale = m.load_keyframes(self.rows, self.identity)
        self.assertEqual(restored, anchors); self.assertFalse(stale)
        Image.new('RGB', (100, 80), 'red').save(self.images[3])
        rows = m.load_sequence(self.images, self.paths, self.identity)
        restored, stale = m.load_keyframes(rows, self.identity)
        self.assertNotIn(3, restored); self.assertTrue(stale)

    def test_bidirectional_disagreement_and_lost_object(self):
        result = m.merge_directions({0: [0, 0, 10, 10], 1: None}, {0: [50, 50, 60, 60], 1: None}, 0, 1)
        self.assertIn('双向跟踪分歧', result[0]['reasons'])
        self.assertIsNone(result[1]['box'])

    def test_external_manual_change_invalidates_saved_anchor(self):
        m.save_keyframes(self.rows, self.identity, self.anchors)
        doc = json.loads(self.paths[0].read_text(encoding='utf-8'))
        doc['shapes'][0]['points'] = [[11, 11], [31, 11], [31, 31], [11, 31]]
        self.paths[0].write_text(json.dumps(doc), encoding='utf-8')
        rows = m.load_sequence(self.images, self.paths, self.identity)
        restored, stale = m.load_keyframes(rows, self.identity)
        self.assertNotIn(0, restored)
        self.assertIn('00001.jpg', stale)

    def test_reference_disagreement_is_a_review_prompt_not_a_confidence(self):
        pred = copy.deepcopy(self.predictions)
        pred[6]['box'] = [50, 40, 70, 60]
        result = m.review_predictions(pred, {0: self.anchors[0]}, self.rows)
        self.assertIn('与已有框差异较大', result[6]['reasons'])

    def test_saved_preview_requires_matching_inputs_and_human_keyframes(self):
        m.save_draft(self.rows, self.identity, self.anchors, self.predictions, (0, 6))
        restored, frame_range = m.load_draft(self.rows, self.identity, self.anchors)
        self.assertEqual(restored, self.predictions)
        self.assertEqual(frame_range, (0, 6))
        changed = copy.deepcopy(self.anchors); changed[0][0] += 1
        self.assertIsNone(m.load_draft(self.rows, self.identity, changed))
        self.paths[3].write_text('{}')
        rows = copy.deepcopy(self.rows); rows[3]['raw'] = b'{}'
        self.assertIsNone(m.load_draft(rows, self.identity, self.anchors))

    def test_soft_rotor_edges_require_foreground_and_connection(self):
        z = np.full((30, 40), -8.)
        z[12:18, 15:25] = 3.
        z[11, 17:23] = -1.
        z[1:3, 1:3] = -.5
        self.assertEqual(mask_box(z), [15., 12., 25., 18.])
        self.assertEqual(mask_box(z, soft=True), [15., 11., 25., 18.])
        self.assertIsNone(mask_box(np.full((30, 40), -1.), soft=True))

    def test_human_margins_are_per_edge_scaled_and_bounded(self):
        margins = learn_edge_margins([([8, 9, 32, 33], [10, 10, 30, 30])])
        expanded = apply_edge_margins([20, 20, 60, 60], margins, (100, 80))
        self.assertLess(expanded[0], 20)
        self.assertGreater(expanded[2], 60)
        self.assertNotEqual(20-expanded[0], 20-expanded[1])
        self.assertEqual(learn_edge_margins([([0, 0, 100, 80], [10, 10, 30, 30])])['relative'], [0.]*4)
        self.assertIsNone(apply_edge_margins(None, margins, (100, 80)))

    def test_rotor_protection_retains_thin_original_edges_but_not_bad_large_boxes(self):
        original = [826.965, 369.005, 939.602, 441.443]
        box, protected = preserve_nearby_outer_edges([828, 373, 936, 440], original)
        self.assertTrue(protected)
        self.assertEqual(box, original)
        unchanged, protected = preserve_nearby_outer_edges([10, 10, 30, 30], [0, 0, 100, 80])
        self.assertFalse(protected); self.assertEqual(unchanged, [10, 10, 30, 30])
        self.assertEqual(preserve_nearby_outer_edges(None, original), (None, False))
        # Padding alone must not turn an already-correct old box into a looser
        # one. Actual image evidence outside it may still expand that edge.
        padded = [824, 367, 942, 444]
        kept, _ = preserve_nearby_outer_edges(padded, original, [828, 373, 936, 440])
        self.assertEqual(kept, original)
        observed, _ = preserve_nearby_outer_edges(padded, original, [824, 373, 936, 440])
        self.assertEqual(observed[0], 824)

    def test_rotor_mode_invalidates_regular_preview_and_warns_about_small_shrink(self):
        m.save_draft(self.rows, self.identity, self.anchors, self.predictions, (0, 6))
        options = {'boundary_mode': 'rotor'}
        self.assertIsNone(m.load_draft(self.rows, self.identity, self.anchors, options))
        m.save_draft(self.rows, self.identity, self.anchors, self.predictions, (0, 6), options)
        self.assertIsNotNone(m.load_draft(self.rows, self.identity, self.anchors, options))
        path = m.session_path(self.rows, self.identity).with_suffix('.draft.json')
        cache = json.loads(path.read_text(encoding='utf-8'))
        cache['options']['boundary_revision'] = 1
        path.write_text(json.dumps(cache), encoding='utf-8')
        self.assertIsNone(m.load_draft(self.rows, self.identity, self.anchors, options))
        pred = {6: {'box': [12, 12, 28, 28], 'reasons': [], 'boundary_mode': 'rotor'}}
        reviewed = m.review_predictions(pred, {}, self.rows)
        self.assertIn('AI 框内缩，请核对旋翼浅灰边', reviewed[6]['reasons'])
        self.assertEqual(m.saved_tracking_options(self.rows, ('drone', 1))['boundary_mode'], 'rotor')
        m.save_keyframes(self.rows, ('drone', 1), self.anchors, {'boundary_mode': 'mask'})
        self.assertEqual(m.saved_tracking_options(self.rows, ('drone', 1))['boundary_mode'], 'mask')

    def test_bidirectional_rotor_merge_keeps_arm_seen_in_one_direction(self):
        # One direction keeps a thin extension; averaging the coordinates hides it
        # even when overlap stays high.
        forward={11:[839.,699.,899.,727.]}
        backward={11:[853.,699.,898.,726.]}
        regular=m.merge_directions(forward,backward,0,12)
        rotor=m.merge_directions(forward,backward,0,12,preserve_edges=True)
        self.assertGreater(regular[11]['box'][0],850.)
        self.assertEqual(rotor[11]['box'],[839.,699.,899.,727.])
        self.assertTrue(rotor[11]['reasons'])

    def test_local_coordinate_mapping_is_reversible_at_image_edges(self):
        for box in ([0.,0.,22.,12.],[1890.,940.,1920.,960.],[811.25,609.83,896.9,645.39]):
            window=local_crop_window(box,(1920,960))
            self.assertGreaterEqual(min(window),0)
            self.assertLessEqual(window[2],1920)
            self.assertLessEqual(window[3],960)
            mapped=crop_box_coordinates(box,window)
            restored=crop_box_coordinates(mapped,window,inverse=True)
            np.testing.assert_allclose(restored,box,atol=1e-10)

    def test_local_refinement_does_not_revive_lost_target_or_accept_background(self):
        size=(1920,960)
        coarse=[850.,699.,898.,727.]
        box,reasons=combine_local_extent(None,[839.,697.,900.,728.],size)
        self.assertIsNone(box)
        box,reasons=combine_local_extent(coarse,[760.,600.,1050.,850.],size)
        self.assertEqual(box,coarse)
        self.assertTrue(reasons)
        box,reasons=combine_local_extent(coarse,None,size)
        self.assertEqual(box,coarse)
        self.assertTrue(reasons)

    def test_local_image_evidence_can_restore_a_thin_arm_and_requests_review(self):
        box,reasons=combine_local_extent([850.,699.,898.,727.],
                                         [834.,697.,900.,728.],(1920,960))
        self.assertEqual(box,[834.,697.,900.,728.])
        self.assertTrue(reasons)

    def test_human_boundary_warning_survives_review_without_changing_coordinates(self):
        warning='人工框与局部轮廓不一致，请核对外缘'
        predictions={0:{'box':[11.,11.,29.,29.],'reasons':[],
                        'edge_review_reasons':[warning]}}
        reviewed=m.review_predictions(predictions,self.anchors,self.rows)
        self.assertEqual(reviewed[0]['box'],self.anchors[0])
        self.assertIn(warning,reviewed[0]['reasons'])

    def test_previous_rotor_revision_is_not_reused_as_refined_preview(self):
        options={'boundary_mode':'rotor'}
        m.save_draft(self.rows,self.identity,self.anchors,self.predictions,(0,6),options)
        path=m.session_path(self.rows,self.identity).with_suffix('.draft.json')
        cache=json.loads(path.read_text(encoding='utf-8'))
        cache['options']['boundary_revision']=2
        path.write_text(json.dumps(cache),encoding='utf-8')
        self.assertIsNone(m.load_draft(self.rows,self.identity,self.anchors,options))
        self.assertEqual(m.saved_draft_range(self.rows,self.identity,self.anchors),(0,6))

    def test_local_bottom_is_not_dragged_back_into_background_roof(self):
        # Local evidence ends above a coarse box that has taken in extra background.
        local=[934.3,625.8555555556,970.5275,641.3222037037]
        coarse=[935.5,626.5,970.,647.]
        aspect=(647.56-631.3555555556)/(968.025-932.)
        box,reasons=combine_local_extent(coarse,local,(1920,960),aspect)
        self.assertAlmostEqual(box[3],local[3])
        self.assertEqual(box[:3],local[:3])
        self.assertTrue(any('抑制' in reason for reason in reasons))

    def test_aspect_check_keeps_coarse_wing_when_it_matches_human_framing(self):
        coarse=[831.,697.,901.,729.]
        local=[845.,697.,901.,729.]
        box,_=combine_local_extent(coarse,local,(1920,960),.45)
        self.assertEqual(box,coarse)

    def test_aspect_check_does_not_force_a_fixed_human_box_size(self):
        local=[900.,600.,940.,620.]
        coarse=[899.,599.,941.,621.]
        box,_=combine_local_extent(coarse,local,(1920,960),.4)
        self.assertEqual(box,coarse)

    def test_reference_aspect_uses_only_current_interval_human_endpoints(self):
        anchors={0:[0.,0.,100.,10.],5:[20.,20.,60.,40.],10:[30.,30.,70.,50.]}
        reference=interval_reference_aspect(anchors,7,5,10)
        anchors[0]=[0.,0.,100.,100.]
        self.assertEqual(reference,.5)
        self.assertEqual(interval_reference_aspect(anchors,7,5,10),reference)
        self.assertIsNone(interval_reference_aspect({},7,5,10))

    def test_continuous_local_arms_survive_small_coarse_fuselage(self):
        coarse=[936.,750.,951.,765.]
        local=[925.5,750.5,962.,765.]
        observed={0:[927.,745.,961.5,759.5],1:local,2:[926.,755.,963.,770.]}
        self.assertTrue(local_size_supported(observed,1))
        old,_=combine_local_extent(coarse,local,(1920,960),.55)
        new,_=combine_local_extent(coarse,local,(1920,960),.55,True)
        self.assertEqual(old,coarse)
        self.assertLessEqual(new[0],local[0]);self.assertGreaterEqual(new[2],local[2])

    def test_one_bad_local_mask_cannot_use_neighbour_exception(self):
        observed={0:[10,10,30,20],1:[0,0,90,70],2:[11,10,31,20]}
        self.assertFalse(local_size_supported(observed,1))
        box,_=combine_local_extent([10,10,30,20],[0,0,90,70],(100,80),.5,True)
        self.assertEqual(box,[10.,10.,30.,20.])
        self.assertFalse(local_size_supported({0:None,1:[0,0,5,5]},1))

    def test_v3_outer_union_draft_is_invalid_after_background_fix(self):
        options={'boundary_mode':'rotor'}
        m.save_draft(self.rows,self.identity,self.anchors,self.predictions,(0,6),options)
        path=m.session_path(self.rows,self.identity).with_suffix('.draft.json')
        cache=json.loads(path.read_text(encoding='utf-8'))
        cache['options']['boundary_revision']=3
        path.write_text(json.dumps(cache),encoding='utf-8')
        self.assertIsNone(m.load_draft(self.rows,self.identity,self.anchors,options))


    def test_out_of_bounds_source_box_does_not_block_preview_or_get_overwritten(self):
        doc = json.loads(self.paths[0].read_text(encoding='utf8'))
        doc['shapes'][0]['points'] = [[10,-2],[110,-2],[110,30],[10,30]]
        self.paths[0].write_text(json.dumps(doc),encoding='utf8')
        original = self.paths[0].read_bytes()
        rows = m.load_sequence(self.images,self.paths,self.identity)
        self.assertEqual(rows[0]['box'],[10.,0.,100.,30.])
        self.assertIn('超出画面',rows[0]['source_box_issue'])
        predictions = {0:{'box':[11,0,99,30],'reasons':[]}, 1:{'box':[11,10,31,30],'reasons':[]}}
        reviewed = m.review_predictions(predictions,{},rows)
        self.assertTrue(any('超出画面' in r for r in reviewed[0]['reasons']))
        backup, _ = m.apply_draft(rows,reviewed,{},self.identity)
        self.assertEqual(self.paths[0].read_bytes(),original)
        m.undo_tracking(backup,self.root)
        self.assertEqual(self.paths[0].read_bytes(),original)

    def test_completely_outside_source_box_is_kept_without_prediction_write(self):
        doc = json.loads(self.paths[0].read_text(encoding='utf8'))
        doc['shapes'][0]['points'] = [[110,10],[120,10],[120,30],[110,30]]
        self.paths[0].write_text(json.dumps(doc),encoding='utf8')
        rows = m.load_sequence(self.images,self.paths,self.identity)
        self.assertIsNone(rows[0]['box'])
        self.assertIn('完全位于画面外',rows[0]['source_box_issue'])
        self.assertEqual(m.build_writes(rows,{0:{'box':[10,10,30,30]}},{},self.identity),[])


if __name__ == '__main__': unittest.main()

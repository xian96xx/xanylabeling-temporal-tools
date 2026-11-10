"""Human keyframes -> asynchronous AI draft -> local correction -> apply/undo."""
import copy
import json
from pathlib import Path
import uuid

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from xal_temporal.recovery import plan_recovery
from xal_temporal.runtime import worker_script

from xal_temporal import tracking
from xal_temporal import masks as mask_store
from xal_temporal.qt.smoothing_dialog import BoxComparison, EDGE_NAMES


class TemporalTrackingDialog(QtWidgets.QDialog):
    def __init__(self, parent, images, label_paths, identities, current_index=0, selected_identity=None):
        super().__init__(parent)
        self.setWindowTitle('AI 补齐 · 人工关键帧辅助跟踪')
        self.resize(1200, 850)
        self.images, self.label_paths = images, label_paths
        self.current_index = current_index
        self.rows, self.anchors, self.predictions, self.prompts = [], {}, {}, {}
        self.pending_spans, self.last_anchors = [], {}
        self.previous_predictions = {}
        self.recovery_baseline = None
        self.last_preview_snapshot = {}
        self.mask_prompts = {}
        self.busy, self.applied = False, False
        self.process = None
        self.worker_ready = False
        self.inspection = None
        self.inspection_snapshot = None
        self.inspection_apply_enabled = False
        self.inspection_action = None
        self._stroke_original = None
        self.stdout_buffer = b''
        self.display_values, self.numeric_context = [None]*4, {}
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.recompute_timer = QtCore.QTimer(self)
        self.recompute_timer.setSingleShot(True)
        self.recompute_timer.timeout.connect(self.generate_preview)

        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel('画少量关键帧，让 AI 补齐中间帧')
        font = heading.font(); font.setPointSize(font.pointSize()+3); font.setBold(True); heading.setFont(font)
        layout.addWidget(heading)
        description = QtWidgets.QLabel('默认只用当前帧作起点。拖动右侧框边后，该帧成为人工关键帧，稍后自动重算附近区间。预览完成后点击应用才改标注。')
        description.setWordWrap(True); layout.addWidget(description)
        body = QtWidgets.QHBoxLayout()
        controls = QtWidgets.QWidget(); controls.setFixedWidth(285)
        form = QtWidgets.QFormLayout(controls)
        self.target = QtWidgets.QComboBox()
        for identity in identities:
            self.target.addItem(f'ID {identity[1]} · {identity[0]}', identity)
        if selected_identity in identities:
            self.target.setCurrentIndex(identities.index(selected_identity))
        form.addRow('目标', self.target)
        self.start, self.end = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin in (self.start, self.end): spin.setRange(1, len(images))
        self.start.setValue(current_index+1); self.end.setValue(min(len(images), current_index+100))
        form.addRow('起始帧', self.start); form.addRow('结束帧', self.end)
        self.whole = QtWidgets.QPushButton('选择整段'); self.whole.clicked.connect(self.whole_sequence); form.addRow(self.whole)
        self.keyframes = QtWidgets.QLineEdit()
        self.keyframes.setPlaceholderText('例如 1, 40, 80')
        self.keyframes.setToolTip('列表使用当前序列的帧序号。已有框不会自动成为依据；请输入你确认过的帧。人工修正会自动加入。')
        self.keyframes.editingFinished.connect(self.set_keyframes_from_text)
        form.addRow('人工关键帧', self.keyframes)
        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItem('常规轮廓', 'mask')
        self.boundary.addItem('旋翼 / 浅灰边', 'rotor')
        self.boundary.setToolTip('旋翼模式增加目标局部视频精修，并用邻近人工框的比例检查额外扩张，避免全图误检把背景重新并入。姿态变化及轮廓分歧仍需检查。')
        form.addRow('框边界', self.boundary)
        self.boundary.currentIndexChanged.connect(self.boundary_changed)
        self.recovery_mode = QtWidgets.QCheckBox('自动回溯可疑片段')
        self.recovery_mode.setToolTip('旋翼模式下，根据两路框的明显分歧生成回溯候选；候选仍可能漏机臂或包含背景，应用前需检查。')
        form.addRow(self.recovery_mode)
        self.point_mode = QtWidgets.QComboBox()
        self.point_mode.addItem('拖框 / 移动', None)
        self.point_mode.addItem('正点：机臂、桨或脚架', 1)
        self.point_mode.addItem('负点：附近背景', 0)
        self.point_mode.addItem('画笔：补目标区域', 'paint')
        self.point_mode.addItem('画笔：擦除背景', 'erase')
        self.point_mode.setToolTip('在人工关键帧右图放点；画笔需先查看识别区域。黄点为目标，蓝点为背景，绿色区域可补画或擦除。')
        self.point_mode.currentIndexChanged.connect(self.change_point_mode)
        form.addRow('部件点提示', self.point_mode)
        self.clear_points = QtWidgets.QPushButton('清除当前帧的点'); self.clear_points.clicked.connect(self.clear_frame_points); form.addRow(self.clear_points)
        self.inspect_button = QtWidgets.QPushButton('查看关键帧识别区域')
        self.inspect_button.setToolTip('对当前人工关键帧单独运行一次局部分割；绿色是模型认出的区域，不代表邻帧传播结果。')
        self.inspect_button.clicked.connect(self.inspect_keyframe); form.addRow(self.inspect_button)
        self.compare_button = QtWidgets.QPushButton('对比涂抹前后')
        self.compare_button.setToolTip('左边显示模型原始区域，右边显示当前涂抹区域；对比的是关键帧局部分割。')
        self.compare_button.clicked.connect(self.compare_mask); form.addRow(self.compare_button)
        self.compare_coarse = QtWidgets.QCheckBox('显示全图候选（橙色虚线）')
        self.compare_coarse.setToolTip('全图候选可能漏掉细桨或机臂，请对照图像后自行判断。')
        self.compare_coarse.toggled.connect(self.show_frame); form.addRow(self.compare_coarse)
        self.use_coarse = QtWidgets.QPushButton('采用全图候选并设关键帧')
        self.use_coarse.setToolTip('将当前帧的全图候选作为人工关键帧；不会直接写入标注。')
        self.use_coarse.clicked.connect(self.use_coarse_candidate); form.addRow(self.use_coarse)
        self.brush_size = QtWidgets.QSpinBox(); self.brush_size.setRange(1,60); self.brush_size.setValue(6); self.brush_size.setSuffix(' 像素')
        self.clear_mask = QtWidgets.QPushButton('清除本帧涂抹'); self.clear_mask.clicked.connect(self.clear_frame_mask)
        brush_row = QtWidgets.QWidget(); brush_layout = QtWidgets.QHBoxLayout(brush_row); brush_layout.setContentsMargins(0,0,0,0)
        brush_layout.addWidget(self.brush_size); brush_layout.addWidget(self.clear_mask); form.addRow('画笔', brush_row)
        self.generate = QtWidgets.QPushButton('生成 / 更新 AI 预览'); self.generate.clicked.connect(self.generate_preview); form.addRow(self.generate)
        self.cancel = QtWidgets.QPushButton('取消计算'); self.cancel.setEnabled(False); self.cancel.clicked.connect(self.cancel_job); form.addRow(self.cancel)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); form.addRow(self.progress)
        self.follow = QtWidgets.QCheckBox('跟随目标放大'); self.follow.setChecked(True); self.follow.toggled.connect(self.show_frame)
        self.compare_previous = QtWidgets.QCheckBox('对比重算前框'); self.compare_previous.toggled.connect(self.show_frame)
        self.compare_recovery = QtWidgets.QCheckBox('对比回溯前框'); self.compare_recovery.toggled.connect(self.show_frame)
        display_row = QtWidgets.QWidget(); display_layout = QtWidgets.QHBoxLayout(display_row); display_layout.setContentsMargins(0,0,0,0)
        display_layout.addWidget(self.follow); display_layout.addWidget(self.compare_previous); form.addRow(display_row)
        form.addRow(self.compare_recovery)
        self.auto_update = QtWidgets.QCheckBox('修框后自动重算附近区间'); self.auto_update.setChecked(True); form.addRow(self.auto_update)
        self.auto_update.toggled.connect(self.auto_update_changed)
        self.loop_review = QtWidgets.QCheckBox('循环播放选中的待检查段'); self.loop_review.setChecked(True); form.addRow(self.loop_review)
        self.review_range = None
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True); form.addRow(self.hint)
        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.controls_scroll.setFixedWidth(305)
        self.controls_scroll.setMinimumHeight(0)
        self.controls_scroll.setWidget(controls)
        body.addWidget(self.controls_scroll)
        view = QtWidgets.QVBoxLayout()
        self.comparison = BoxComparison(); self.comparison.setMinimumSize(360,160)
        self.comparison.edgeEdited.connect(self.edit_edge); self.comparison.editStarted.connect(self.begin_edit)
        self.comparison.allow_move = True
        self.comparison.boxEdited.connect(self.edit_box)
        self.comparison.pointAdded.connect(self.add_prompt_point)
        self.comparison.maskStroke.connect(self.paint_mask_stroke)
        self.comparison.maskStrokeFinished.connect(self.finish_mask_stroke)
        view.addWidget(self.comparison, 1)
        navigation = QtWidgets.QHBoxLayout()
        self.frame_info = QtWidgets.QLabel(); self.frame_info.setWordWrap(True); navigation.addWidget(self.frame_info, 1)
        previous_frame = QtWidgets.QPushButton('上一帧'); previous_frame.clicked.connect(lambda: self.step_frame(-1)); navigation.addWidget(previous_frame)
        self.seek = QtWidgets.QSpinBox(); self.seek.setRange(1, len(images)); self.seek.setPrefix('第 '); self.seek.setSuffix(' 帧'); self.seek.setKeyboardTracking(False)
        self.seek.valueChanged.connect(lambda value: self.slider.setValue(value-1)); navigation.addWidget(self.seek)
        next_frame = QtWidgets.QPushButton('下一帧'); next_frame.clicked.connect(lambda: self.step_frame(1)); navigation.addWidget(next_frame)
        view.addLayout(navigation)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider.valueChanged.connect(self.show_frame); view.addWidget(self.slider)
        editors = QtWidgets.QHBoxLayout(); self.edge_values = []
        for edge, name in enumerate(EDGE_NAMES):
            editors.addWidget(QtWidgets.QLabel(name))
            spin = QtWidgets.QDoubleSpinBox(); spin.setDecimals(2); spin.setKeyboardTracking(False)
            spin.setSingleStep(.5); spin.setAccessibleName(name+'边坐标')
            spin.editingFinished.connect(lambda e=edge: self.finish_numeric_edit(e))
            spin.installEventFilter(self); spin.lineEdit().installEventFilter(self)
            editors.addWidget(spin); self.edge_values.append(spin)
        view.addLayout(editors)
        playback = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton('播放'); self.play.clicked.connect(self.toggle_play); playback.addWidget(self.play)
        self.pin = QtWidgets.QPushButton('确认当前框为关键帧'); self.pin.clicked.connect(self.pin_frame); playback.addWidget(self.pin)
        self.unpin = QtWidgets.QPushButton('取消当前关键帧'); self.unpin.clicked.connect(self.unpin_frame); playback.addWidget(self.unpin)
        self.previous = QtWidgets.QPushButton('上一处待检查'); self.previous.clicked.connect(lambda: self.jump_review(-1)); playback.addWidget(self.previous)
        self.next = QtWidgets.QPushButton('下一处待检查'); self.next.clicked.connect(lambda: self.jump_review(1)); playback.addWidget(self.next)
        self.fps = QtWidgets.QSpinBox(); self.fps.setRange(1, 30); self.fps.setValue(12); self.fps.setSuffix(' 帧/秒'); playback.addWidget(self.fps)
        view.addLayout(playback)
        self.review_list = QtWidgets.QListWidget(); self.review_list.setFixedHeight(64)
        self.review_list.itemClicked.connect(self.select_review)
        view.addWidget(self.review_list)
        body.addLayout(view, 1); layout.addLayout(body, 1)
        self.status = QtWidgets.QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.undo = QtWidgets.QPushButton('撤销上次 AI 补齐'); self.undo.clicked.connect(self.undo_last); buttons.addWidget(self.undo)
        buttons.addStretch()
        close = QtWidgets.QPushButton('关闭'); close.clicked.connect(self.reject); buttons.addWidget(close)
        self.apply = QtWidgets.QPushButton('应用预览'); self.apply.setEnabled(False); self.apply.clicked.connect(self.apply_changes); buttons.addWidget(self.apply)
        layout.addLayout(buttons)
        self.target.currentIndexChanged.connect(self.load_target)
        self.start.valueChanged.connect(self.scope_changed); self.end.valueChanged.connect(self.scope_changed)
        for button in self.findChildren(QtWidgets.QPushButton): button.setAutoDefault(False); button.setDefault(False)
        self.load_target()
        # A tall control panel must not push the action buttons under the taskbar.
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(min(1200, max(1, area.width()-24)), min(850, max(1, area.height()-48)))

    @property
    def identity(self):
        return tuple(self.target.currentData())

    @property
    def options(self):
        return tracking.tracking_options({'boundary_mode': self.boundary.currentData()})

    def boundary_changed(self, *_):
        self.recompute_timer.stop(); self.pause()
        self.predictions = {}; self.pending_spans = []; self.apply.setEnabled(False)
        self.review_list.clear(); self.review_range = None
        rotor = self.boundary.currentData() == 'rotor'
        self.hint.setText('金色框边：人工关键帧。\n待检查提示不代表准确率。\n'+(
            '旋翼外缘增强：局部精修＋人工框比例校验，抑制额外背景；轮廓分歧需检查。' if rotor else
            '常规模式依据可见轮廓；遮挡、完整车身框和跨接缝目标需重点检查。'))
        if self.rows:
            self.save_anchors(); self.show_frame()
            self.status.setText('框边界模式已改变，请重新生成 AI 预览。')

    def pause(self):
        self.timer.stop(); self.play.setText('播放')

    def load_target(self, *_):
        self.inspection = None
        self.previous_predictions = {}
        self.last_preview_snapshot = {}
        self.pause(); self.recompute_timer.stop()
        try:
            self.rows = tracking.load_sequence(self.images, self.label_paths, self.identity)
            self.anchors, stale = tracking.load_keyframes(self.rows, self.identity)
            self.prompts = tracking.load_prompts(self.rows, self.identity, self.anchors)
            self.mask_prompts = mask_store.load_masks(self.rows, self.identity, self.anchors, self.prompts)
            options = tracking.saved_tracking_options(self.rows, self.identity)
            blocker = QtCore.QSignalBlocker(self.boundary)
            self.boundary.setCurrentIndex(self.boundary.findData(options['boundary_mode'])); del blocker
            self.hint.setText('金色框边：人工关键帧。\n待检查提示不代表准确率。\n'+(
                '旋翼外缘增强：局部精修＋人工框比例校验，抑制额外背景；轮廓分歧需检查。' if options['boundary_mode'] == 'rotor' else
                '常规模式依据可见轮廓；遮挡、完整车身框和跨接缝目标需重点检查。'))
            # Reopening on a human frame edited outside this dialog must use
            # its current saved box even when other cached anchors survive.
            current_was_changed = self.rows[self.current_index]['image'].name in stale
            cached = tracking.load_draft(self.rows, self.identity, self.anchors, options, self.prompts, self.mask_prompts)
            # A valid batch draft may have assistant prompts but no confirmed
            # human frames. Merely opening it must not promote an AI box.
            if (cached is None and (not self.anchors or current_was_changed)
                    and self.rows[self.current_index]['box'] is not None):
                self.anchors[self.current_index] = list(self.rows[self.current_index]['box'])
            self.predictions, self.last_anchors, self.pending_spans = {}, {}, []
            self.sync_keyframes(); self.scope_changed()
            self.slider.setValue(self.current_index)
            self.undo.setEnabled(tracking.latest_tracking_backup(self.rows[0]['path'].parent) is not None)
            source_issue_count = sum(bool(row.get('source_box_issue')) for row in self.rows)
            self.status.setText(f'已有 {len(self.anchors)} 个人工关键帧；其余已有框只用于对照。'+
                                (f' {source_issue_count} 帧原框越界或无效：仅预览可见区域，原标注不自动覆盖。' if source_issue_count else '')+
                                (f' {len(stale)} 条旧关键帧因图片或标注变化未载入。' if stale else ''))
            cached = tracking.load_draft(self.rows, self.identity, self.anchors, self.options, self.prompts, self.mask_prompts)
            if cached is not None:
                self.predictions, frame_range = cached
                self.last_anchors = copy.deepcopy(self.anchors)
                self.start.setValue(frame_range[0]+1); self.end.setValue(frame_range[1]+1)
                self.predictions = tracking.review_predictions(self.predictions, self.anchors, self.rows)
                self.last_preview_snapshot = copy.deepcopy(self.predictions)
                self.refresh_reviews(); self.show_frame(); self.apply.setEnabled(True)
                self.progress.setRange(0, 100); self.progress.setValue(100)
                matches_saved = all(
                    (item.get('box') is None and self.rows[i]['box'] is None) or
                    (item.get('box') is not None and self.rows[i]['box'] is not None and
                     np.allclose(item['box'],self.rows[i]['box'],rtol=0,atol=1e-8))
                    for i,item in self.predictions.items())
                self.apply.setEnabled(not matches_saved)
                self.status.setText(
                    f'已恢复第 {frame_range[0]+1}–{frame_range[1]+1} 帧的 AI 结果，无需重新计算。'+
                    ('这些框已写入标注，可直接检查。' if matches_saved else '草稿中的改动尚未写入标注。'))
            elif options['boundary_mode'] == 'rotor':
                old_range = tracking.saved_draft_range(self.rows, self.identity, self.anchors)
                if old_range is not None:
                    self.start.setValue(old_range[0]+1); self.end.setValue(old_range[1]+1)
                self.status.setText(f'已保留 {len(self.anchors)} 个人工关键帧；已选「旋翼 / 浅灰边」，请重新生成外缘增强预览。旧版本草稿不会套用。')
        except Exception as error:
            self.rows = []; self.status.setText(str(error)); self.generate.setEnabled(False)

    def sync_keyframes(self):
        self.keyframes.setText(', '.join(str(i+1) for i in sorted(self.anchors)))

    def scope_changed(self, *_):
        self.pause(); self.recompute_timer.stop(); self.pending_spans = []
        start, end = self.start.value()-1, self.end.value()-1
        if end >= start:
            self.slider.setRange(start, end)
        self.apply.setEnabled(False); self.show_frame()

    def whole_sequence(self):
        self.start.setValue(1); self.end.setValue(len(self.images))

    def save_anchors(self):
        tracking.save_keyframes(self.rows, self.identity, self.anchors, self.options)

    def set_keyframes_from_text(self):
        if not self.rows or self.busy: return
        try:
            text = self.keyframes.text().replace('，', ',').replace(',', ' ')
            keys = {int(v)-1 for v in text.split()}
            if any(i < 0 or i >= len(self.rows) for i in keys):
                raise ValueError('关键帧序号超出序列。')
            changed = keys.symmetric_difference(self.anchors)
            updated = {}
            for i in keys:
                box = self.anchors.get(i, self.rows[i]['box'])
                if box is None:
                    raise ValueError(f'第 {i+1} 帧没有原框，请先在预览里修正并确认。')
                updated[i] = list(box)
            self.anchors = updated
            self.prompts = {i:p for i,p in self.prompts.items() if i in self.anchors}
            self.mask_prompts = mask_store.load_masks(self.rows, self.identity, self.anchors, self.prompts)
            tracking.save_prompts(self.rows, self.identity, self.anchors, self.prompts)
            for i in changed: self.mark_pending(i)
            self.save_anchors(); self.sync_keyframes(); self.show_frame()
        except Exception as error:
            self.status.setText(str(error)); self.sync_keyframes()

    def mark_pending(self, frame):
        if self.boundary.currentData() == 'rotor' and self.last_anchors and self.anchors:
            first, previous_first = min(self.anchors), min(self.last_anchors)
            if first != previous_first or frame == first:
                # The initial example defines the common blur/rotor margin.
                # Changing that example requires refreshing the whole scope.
                self.predictions = {}
        self.pending_spans.append(tracking.affected_span(self.anchors, frame, self.start.value()-1, self.end.value()-1))
        self.apply.setEnabled(False)

    def show_frame(self, *_):
        if not self.rows: return
        i = self.slider.value(); row = self.rows[i]
        blocker = QtCore.QSignalBlocker(self.seek); self.seek.setValue(i+1); del blocker
        item = self.predictions.get(i, {})
        candidate = item.get('coarse_evidence_box') if i not in self.anchors else None
        try:
            candidate = tracking.valid_box(candidate, row['size']) if candidate is not None else None
        except (ValueError,TypeError):
            candidate = None
        box = self.anchors.get(i, item.get('box'))
        context = box if box is not None else row['box']
        if context is None:
            context = next(iter(self.anchors.values()), [0, 0, row['size'][0], row['size'][1]])
        original = row['box'] if row['box'] is not None else context
        previous = item.get('pre_recovery_box') if self.compare_recovery.isChecked() else None
        if previous is None and self.compare_previous.isChecked(): previous = self.previous_predictions.get(i, {}).get('box')
        if previous is not None: original = previous
        new = np.asarray(box if box is not None else context, dtype=float)
        display = {'image': row['image'], 'doc': row['doc'], 'box': np.asarray(original),
                   'new_box': new, 'baseline_box': new.copy(), 'fixed_edges': np.array([i in self.anchors]*4),
                   'old_visible': previous is not None or row['box'] is not None, 'new_visible': box is not None,
                   'prompt_points': self.prompts.get(i, []),
                   'coarse_candidate': candidate if self.compare_coarse.isChecked() else None,
                   'inspection_overlay': self.inspection_overlay(i),
                   'baseline_overlay': self.inspection.get('baseline') if self.inspection_overlay(i) and self.inspection else None,
                   'left_title': ('原框越界 · 仅显示可见部分' if row.get('source_box_issue') else '原有框') if row['box'] is not None else ('原框不可见 · 保留原标注' if row.get('source_box_issue') else '原帧 · 尚无目标框'),
                   'right_title': '人工关键帧 · 拖边修正' if i in self.anchors else (
                       'AI 预览 · 拖边修正' if box is not None else '尚无 AI 框 · 可用坐标建立关键帧')}
        if previous is not None: display['left_title'] = '回溯前的 AI 预览框' if self.compare_recovery.isChecked() and item.get('pre_recovery_box') is not None else '重算前的 AI 预览框'
        if display['baseline_overlay']:
            display['left_title'] = '重算前框 / 模型原始区域' if previous is not None else '模型原始局部分割'
        self.comparison.setEnabled(not self.busy)
        crop = QtCore.QRectF(0, 0, *row['size'])
        self.comparison.show_row(display, crop, self.follow.isChecked())
        self.change_point_mode()
        reasons = item.get('reasons', [])
        self.frame_info.setText(f'第 {i+1} / {len(self.rows)} 帧 · {row["image"].name}'+
                               (' · 人工固定' if i in self.anchors else '')+
                               ('\n'+row['source_box_issue'] if row.get('source_box_issue') else '')+
                               (f' · 部件提示 {len(self.prompts[i])} 点' if i in self.prompts else '')+
                               (' · 已保存涂抹' if i in self.mask_prompts else '')+
                               ('\n绿色：关键帧局部分割及人工涂抹；不代表邻帧传播结果。' if display['inspection_overlay'] else '')+
                               ('\n待检查：'+'；'.join(reasons) if reasons else '')+
                               ('\n'+'；'.join(item.get('notices', [])) if item.get('notices') else ''))
        for edge, spin in enumerate(self.edge_values):
            blocker = QtCore.QSignalBlocker(spin)
            low = new[edge-2]+.01 if edge in (2, 3) else 0
            high = new[edge+2]-.01 if edge in (0, 1) else row['size'][edge-2]
            spin.setRange(float(low), float(high)); spin.setValue(float(new[edge])); spin.setEnabled(not self.busy)
            self.display_values[edge] = spin.value(); del blocker
        self.pin.setEnabled(not self.busy and (box is not None or row['box'] is not None))
        self.unpin.setEnabled(not self.busy and i in self.anchors)
        self.use_coarse.setEnabled(not self.busy and candidate is not None and self.compare_coarse.isChecked())
        self.inspect_button.setEnabled(not self.busy and i in self.anchors)
        self.compare_button.setEnabled(not self.busy and i in self.mask_prompts and bool(display['inspection_overlay']))
        self.inspect_button.setText('隐藏关键帧识别区域' if display['inspection_overlay'] else '查看关键帧识别区域')

    def use_coarse_candidate(self):
        if self.busy or not self.rows:return
        frame = self.slider.value()
        candidate = self.predictions.get(frame,{}).get('coarse_evidence_box')
        if frame in self.anchors or candidate is None or not self.compare_coarse.isChecked():return
        try:
            self.anchors[frame] = tracking.valid_box(candidate,self.rows[frame]['size'])
            self.anchor_changed(frame)
            self.status.setText(f'第 {frame+1} 帧已采用全图候选并设为人工关键帧；请检查细桨，然后重新生成附近预览。')
        except Exception as error:self.status.setText(str(error))

    def inspection_overlay(self, frame):
        item = self.inspection
        if not item or item['frame'] != frame or item['identity'] != self.identity:
            return None
        if self.anchors.get(frame) != item['box'] or self.prompts.get(frame, []) != item['points']:
            return None
        try:
            stamp = self.rows[frame]['image'].stat()
            if [stamp.st_size,stamp.st_mtime_ns] != item['image_stamp']: return None
        except OSError:
            return None
        return item.get('image') if item.get('image') is not None else None

    def inspect_keyframe(self):
        if self.busy or not self.rows: return
        frame = self.slider.value()
        if self.inspection_overlay(frame):
            self.inspection = None; self.show_frame()
            self.status.setText('已隐藏识别区域。')
            return
        if frame not in self.anchors:
            self.status.setText('请先切换到已确认的人工关键帧，再查看识别区域。')
            return
        if frame in self.mask_prompts:
            from xal_temporal.geometry import local_crop_window
            record = self.mask_prompts[frame]
            image = QtGui.QImage(record['path'])
            if not image.isNull():
                self.inspection = {'frame':frame,'identity':self.identity,'box':copy.deepcopy(self.anchors[frame]),
                    'points':copy.deepcopy(self.prompts.get(frame,[])), 'image_stamp':list(self.rows[frame]['image_stamp']),
                    'overlay':record['path'], 'image':image,
                    'window':local_crop_window(self.anchors[frame],self.rows[frame]['size'])}
                self.show_frame(); self.status.setText('显示已保存的关键帧涂抹，可继续补画或擦除。')
                return
        try:
            self.pause(); self.recompute_timer.stop()
            config = tracking.runtime_config()
            folder = tracking.PROJECT_ROOT/'runtime/tracking_jobs'/uuid.uuid4().hex
            folder.mkdir(parents=True)
            self.jobfile = folder/'job.json'
            self.inspection_snapshot = {'frame':frame,'identity':self.identity,
                'box':copy.deepcopy(self.anchors[frame]), 'points':copy.deepcopy(self.prompts.get(frame, [])),
                'image_stamp':list(self.rows[frame]['image_stamp'])}
            job = {'operation':'inspect_keyframe', 'image':str(self.rows[frame]['image']),
                   'box':self.inspection_snapshot['box'], 'points':self.inspection_snapshot['points']}
            self.jobfile.write_text(json.dumps(job,ensure_ascii=False),encoding='utf8')
            self.inspection_apply_enabled = self.apply.isEnabled()
            self.inspection_action = 'inspect'
            self.set_busy(True); self.apply.setEnabled(False); self.progress.setRange(0,0)
            self.status.setText('正在检查当前关键帧，绿色将显示模型局部识别区域。')
            self.start_worker(config)
        except Exception as error: self.fail(str(error))

    def compare_mask(self):
        if self.busy or not self.rows:return
        frame = self.slider.value()
        if frame not in self.mask_prompts or not self.inspection_overlay(frame):
            self.status.setText('请先打开关键帧的涂抹区域。'); return
        try:
            self.pause(); self.recompute_timer.stop(); config=tracking.runtime_config()
            folder=tracking.PROJECT_ROOT/'runtime/tracking_jobs'/uuid.uuid4().hex;folder.mkdir(parents=True)
            self.jobfile=folder/'job.json'
            self.inspection_snapshot={'frame':frame,'identity':self.identity,'box':copy.deepcopy(self.anchors[frame]),
                'points':copy.deepcopy(self.prompts.get(frame,[])), 'image_stamp':list(self.rows[frame]['image_stamp']),
                'mask_sha256':self.mask_prompts[frame]['sha256']}
            job={'operation':'inspect_keyframe','image':str(self.rows[frame]['image']),
                 'box':self.inspection_snapshot['box'],'points':self.inspection_snapshot['points']}
            self.jobfile.write_text(json.dumps(job,ensure_ascii=False),encoding='utf8')
            self.inspection_apply_enabled=self.apply.isEnabled(); self.inspection_action='compare'
            self.set_busy(True);self.apply.setEnabled(False);self.progress.setRange(0,0)
            self.status.setText('正在计算模型原始区域，用于与涂抹结果对照。'); self.start_worker(config)
        except Exception as error:self.fail(str(error))

    def accept_inspection(self, event):
        try:
            snap = self.inspection_snapshot
            if snap is None: raise ValueError('关键帧检查已失效，请重新检查。')
            overlay = Path(event['overlay']).resolve()
            if overlay.parent != self.jobfile.parent.resolve() or not overlay.is_file():
                raise ValueError('识别区域文件无效。')
            image = QtGui.QImage(str(overlay))
            if image.isNull(): raise ValueError('无法读取识别区域。')
            if self.inspection_action == 'compare':
                frame=snap['frame']
                if (not self.inspection_overlay(frame) or self.mask_prompts.get(frame,{}).get('sha256')!=snap['mask_sha256']):
                    raise ValueError('涂抹或关键帧已变化，请重新对照。')
                self.inspection['baseline']=image
            else:
                self.inspection = dict(snap, overlay=str(overlay), image=image, window=event['window'])
            self.set_busy(False); self.progress.setRange(0,100); self.progress.setValue(100)
            self.apply.setEnabled(self.inspection_apply_enabled and not self.pending_spans)
            self.show_frame()
            self.status.setText('左图为模型原始识别，右图为当前涂抹；仅供比较关键帧，不代表邻帧结果。' if self.inspection_action == 'compare' else '绿色为当前关键帧局部分割前景，用来检查漏部件或误选背景；不是邻帧传播结果，也没有修改标注。')
        except Exception as error: self.fail(str(error))

    def change_point_mode(self, *_):
        mode = self.point_mode.currentData()
        self.comparison.point_mode = mode if mode in (0,1) and not isinstance(mode,str) else None
        self.comparison.mask_mode = (1 if mode == 'paint' else 0) if mode in ('paint','erase') and self.inspection_overlay(self.slider.value()) else None
        if mode in ('paint','erase') and self.comparison.mask_mode is None:
            self.status.setText('先到人工关键帧点击「查看关键帧识别区域」，再使用画笔。')

    def paint_mask_stroke(self, x0,y0,x1,y1,label):
        frame = self.slider.value()
        if self.busy or not self.inspection_overlay(frame): return
        if self._stroke_original is None: self._stroke_original = self.inspection['image'].copy()
        painter = QtGui.QPainter(self.inspection['image'])
        x,y,x2,y2=self.inspection['window']
        painter.setClipRect(QtCore.QRect(x,y,x2-x,y2-y))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
        color = QtGui.QColor(25,235,130,115) if label else QtGui.QColor(0,0,0,0)
        pen=QtGui.QPen(color,self.brush_size.value(),QtCore.Qt.PenStyle.SolidLine,QtCore.Qt.PenCapStyle.RoundCap,QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if x0 == x1 and y0 == y1: painter.drawPoint(QtCore.QPointF(x0,y0))
        else: painter.drawLine(QtCore.QPointF(x0,y0),QtCore.QPointF(x1,y1))
        painter.end()
        self.comparison.update()

    def finish_mask_stroke(self):
        frame=self.slider.value()
        if not self.inspection_overlay(frame): return
        try:
            buffer=QtCore.QBuffer(); buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
            if not self.inspection['image'].save(buffer,'PNG'): raise ValueError('涂抹区域保存失败。')
            self.save_anchors()
            self.mask_prompts[frame]=mask_store.save_mask(self.rows,self.identity,frame,self.anchors[frame],self.prompts.get(frame,[]),bytes(buffer.data()))
            self._stroke_original = None
            self.mark_pending(frame); self.show_frame()
            self.status.setText('涂抹已保存为当前关键帧提示；点击「生成 / 更新 AI 预览」重新跟踪。原标注未改。')
        except Exception as error:
            if self._stroke_original is not None: self.inspection['image'] = self._stroke_original
            self._stroke_original = None; self.show_frame()
            self.status.setText('涂抹未保存，已恢复上次区域：'+str(error))

    def clear_frame_mask(self):
        if self.busy or not self.rows:return
        frame=self.slider.value()
        if frame not in self.mask_prompts:return
        mask_store.clear_mask(self.rows,self.identity,frame);self.mask_prompts.pop(frame,None)
        self.inspection=None;self.mark_pending(frame);self.show_frame()
        self.status.setText('已清除本帧涂抹提示；请重新生成预览。')

    def add_prompt_point(self, x, y, label):
        if self.busy or not self.rows: return
        i = self.slider.value()
        if i not in self.anchors:
            self.status.setText('部件点请加在已确认的人工关键帧上；不会自动把 AI 框变成关键帧。')
            return
        self.pause(); self.recompute_timer.stop()
        had_mask = i in self.mask_prompts
        self.prompts.setdefault(i, []).append([round(x,2),round(y,2),int(label)])
        tracking.save_prompts(self.rows, self.identity, self.anchors, self.prompts)
        self.mask_prompts = mask_store.load_masks(self.rows, self.identity, self.anchors, self.prompts)
        self.mark_pending(i); self.show_frame()
        self.status.setText(f'第 {i+1} 帧点提示已保存。'+('此前涂抹已暂时停用，请重新查看并涂抹。' if had_mask else '完成放点后点击「生成 / 更新 AI 预览」。黄色是目标，蓝色是背景。'))

    def clear_frame_points(self):
        if self.busy or not self.rows: return
        i = self.slider.value()
        if i not in self.prompts: return
        self.recompute_timer.stop(); self.prompts.pop(i)
        tracking.save_prompts(self.rows, self.identity, self.anchors, self.prompts)
        self.mask_prompts = mask_store.load_masks(self.rows, self.identity, self.anchors, self.prompts)
        self.mark_pending(i); self.show_frame()
        self.status.setText('已清除本帧点提示，请重新生成预览。')

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.FocusIn:
            for edge, spin in enumerate(getattr(self, 'edge_values', [])):
                if obj in (spin, spin.lineEdit()):
                    self.begin_edit(); self.numeric_context[edge] = (self.identity, self.slider.value())
        elif event.type() == QtCore.QEvent.Type.FocusOut:
            if self.pending_spans and self.auto_update.isChecked() and not self.busy:
                self.recompute_timer.start(900)
        return super().eventFilter(obj, event)

    def begin_edit(self):
        self.pause(); self.recompute_timer.stop()

    def auto_update_changed(self, enabled):
        if not enabled:
            self.recompute_timer.stop()
        elif self.pending_spans and not self.busy:
            self.recompute_timer.start(900)

    def finish_numeric_edit(self, edge):
        if self.busy or not self.rows: return
        context = (self.identity, self.slider.value())
        if self.numeric_context.get(edge, context) != context:
            self.show_frame(); return
        value = self.edge_values[edge].value()
        if abs(value-self.display_values[edge]) > 1e-8: self.edit_edge(edge, value)

    def edit_edge(self, edge, value):
        if self.busy or not self.rows: return
        i = self.slider.value()
        box = self.comparison.row['new_box'].copy()
        if abs(box[edge]-value) < 1e-8: return
        box[edge] = value
        try:
            self.anchors[i] = tracking.valid_box(box, self.rows[i]['size'])
            self.anchor_changed(i)
        except Exception as error:
            self.status.setText(str(error)); self.show_frame()

    def anchor_changed(self, frame):
        self.pause(); self.prompts.pop(frame, None)
        self.mask_prompts.pop(frame, None)
        tracking.save_prompts(self.rows, self.identity, self.anchors, self.prompts)
        self.mark_pending(frame); self.save_anchors(); self.sync_keyframes()
        self.predictions[frame] = {'box': list(self.anchors[frame]), 'reasons': []}
        self.show_frame()
        a, b = self.pending_spans[-1]
        self.status.setText(f'第 {frame+1} 帧已保存为人工关键帧；待重算第 {a+1}–{b+1} 帧。')
        if self.auto_update.isChecked(): self.recompute_timer.start(900)

    def edit_box(self, box):
        if self.busy or not self.rows: return
        i = self.slider.value()
        if np.max(np.abs(np.asarray(box)-self.comparison.row['new_box'])) < 1e-7: return
        try:
            self.anchors[i] = tracking.valid_box(box, self.rows[i]['size'])
            self.anchor_changed(i)
        except Exception as error:
            self.status.setText(str(error)); self.show_frame()

    def pin_frame(self):
        i = self.slider.value()
        box = self.comparison.row['new_box'].tolist()
        if i in self.anchors and self.anchors[i] == box:
            self.status.setText(f'第 {i+1} 帧已经是人工关键帧。'); return
        self.anchors[i] = box; self.anchor_changed(i)

    def unpin_frame(self):
        i = self.slider.value()
        if len(self.anchors) <= 1:
            self.status.setText('至少保留一个人工关键帧。'); return
        self.anchors.pop(i, None); self.prompts.pop(i, None); self.mask_prompts.pop(i, None)
        tracking.save_prompts(self.rows, self.identity, self.anchors, self.prompts)
        self.mark_pending(i); self.save_anchors(); self.sync_keyframes(); self.show_frame()
        if self.auto_update.isChecked(): self.recompute_timer.start(900)

    def set_busy(self, busy):
        self.busy = busy; self.pause(); self.cancel.setEnabled(busy)
        for widget in (self.target, self.start, self.end, self.whole, self.keyframes, self.boundary, self.recovery_mode, self.point_mode, self.clear_points, self.brush_size, self.clear_mask, self.generate): widget.setEnabled(not busy)
        self.undo.setEnabled(not busy and bool(self.rows) and tracking.latest_tracking_backup(self.rows[0]['path'].parent) is not None)
        self.show_frame()

    def generate_preview(self):
        if self.busy or not self.rows: return
        self.recompute_timer.stop()
        self.recovery_baseline = None
        try:
            config = tracking.runtime_config()
            start, end = self.start.value()-1, self.end.value()-1
            if end < start: raise ValueError('结束帧不能早于起始帧。')
            intervals = tracking.tracking_intervals(self.anchors, start, end)
            complete = all(i in self.predictions for i in range(start, end+1))
            if complete and self.pending_spans:
                intervals = [(a, b) for a, b in intervals if any(max(a, c) < min(b, d) for c, d in self.pending_spans)]
            self.save_anchors()
            folder = tracking.PROJECT_ROOT/'runtime/tracking_jobs'/uuid.uuid4().hex
            folder.mkdir(parents=True)
            self.jobfile = folder/'job.json'
            self.job_range = (start, end)
            job = {'images': [str(r['image']) for r in self.rows], 'anchors': {
                str(i): b for i, b in self.anchors.items() if start <= i <= end}, 'segments': intervals,
                'mask_prompts': {str(i): p for i,p in self.mask_prompts.items() if start <= i <= end and i in self.anchors},
                'options': self.options, 'prompts': {str(i): p for i,p in self.prompts.items() if start <= i <= end and i in self.anchors}, 'original_boxes': {str(i): r['box'] for i, r in enumerate(self.rows)
                    if r['box'] is not None and start <= i <= end}}
            self.jobfile.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
            self.set_busy(True); self.apply.setEnabled(False); self.progress.setRange(0, 0)
            self.status.setText('正在准备 AI；第一次加载模型需要稍等。')
            self.start_worker(config)
        except Exception as error:
            self.fail(str(error))

    def start_worker(self, config):
        if self.process is None or self.process.state() == QtCore.QProcess.ProcessState.NotRunning:
            self.worker_ready = False; self.stdout_buffer = b''
            self.process = QtCore.QProcess(self)
            env = QtCore.QProcessEnvironment.systemEnvironment(); env.insert('PYTHONUTF8', '1'); env.insert('PYTHONIOENCODING', 'utf-8')
            env.insert('XAL_TEMPORAL_HOME', str(tracking.PROJECT_ROOT))
            self.process.setProcessEnvironment(env)
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.readyReadStandardError.connect(self.read_errors)
            self.process.finished.connect(self.worker_finished)
            self.process.errorOccurred.connect(lambda *_: self.fail('AI 后台进程无法启动，请检查运行环境。'))
            self.process.start(config['python'], ['-I', '-B', '-X', 'utf8', '-u', str(worker_script())])
        elif self.worker_ready:
            self.send_job()

    def send_job(self):
        if self.busy:
            self.process.write((json.dumps({'job': str(self.jobfile)}, ensure_ascii=False)+'\n').encode('utf-8'))

    def read_errors(self):
        data = bytes(self.process.readAllStandardError())
        if hasattr(self, 'jobfile'):
            with (self.jobfile.parent/'worker.log').open('ab') as f: f.write(data)

    def read_output(self):
        self.stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split(b'\n', 1)
            try: event = json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeError): continue
            kind = event.get('type')
            if kind == 'ready':
                self.worker_ready = True; self.send_job()
            elif kind == 'loading': self.status.setText(event['message'])
            elif kind == 'progress':
                self.progress.setRange(0, max(1, event['total'])); self.progress.setValue(event['value']); self.status.setText(event['message'])
            elif kind == 'error': self.fail(event['message'])
            elif kind == 'done': self.accept_result(event)
            elif kind == 'inspection_done': self.accept_inspection(event)

    def accept_result(self, event):
        try:
            result = json.loads(Path(event['result']).read_text(encoding='utf-8'))
            incoming = {int(i): item for i, item in result['predictions'].items()}
            if self.recovery_baseline is not None:
                baseline = self.recovery_baseline; self.recovery_baseline = None
                for i, item in incoming.items():
                    if i in self.anchors or i not in baseline or item.get('box') is None: continue
                    original = baseline[i]
                    item['pre_recovery_box'] = copy.deepcopy(original.get('box'))
                    item['reasons'] = list(dict.fromkeys(original.get('reasons', []) + item.get('reasons', []) + ['自动回溯候选，请检查机臂和背景']))
                    baseline[i] = item
                incoming = baseline
            elif self.recovery_mode.isChecked() and self.options.get('boundary_mode') == 'rotor':
                start, end = self.job_range
                segments, seeds = plan_recovery(incoming, self.anchors, start, end)
                if segments:
                    folder = tracking.PROJECT_ROOT/'runtime/tracking_jobs'/uuid.uuid4().hex
                    folder.mkdir(parents=True); self.jobfile = folder/'job.json'
                    recovery = {'images': [str(r['image']) for r in self.rows], 'anchors': {str(i): b for i,b in seeds.items()}, 'segments': segments,
                                'mask_prompts': {}, 'prompts': {}, 'options': self.options, 'original_boxes': {}}
                    self.jobfile.write_text(json.dumps(recovery, ensure_ascii=False), encoding='utf-8')
                    self.recovery_baseline = incoming
                    self.status.setText(f'正在回溯 {len(segments)} 段可疑区域；结果仍需核对。')
                    self.send_job(); return
            self.previous_predictions = copy.deepcopy(self.last_preview_snapshot)
            self.predictions.update(incoming)
            self.predictions = tracking.review_predictions(self.predictions, self.anchors, self.rows)
            self.last_preview_snapshot = copy.deepcopy(self.predictions)
            self.last_anchors = copy.deepcopy(self.anchors); self.pending_spans = []
            tracking.save_draft(self.rows, self.identity, self.anchors, self.predictions, self.job_range, self.options, self.prompts, self.mask_prompts)
            self.set_busy(False); self.progress.setRange(0, 100); self.progress.setValue(100)
            self.refresh_reviews(); self.apply.setEnabled(True)
            a, b = self.job_range
            missing = sum(self.predictions.get(i, {}).get('box') is None for i in range(a, b+1))
            count = sum(bool(self.predictions.get(i, {}).get('reasons')) for i in range(a, b+1))
            reused = result.get('cached_frames', 0)
            cache_note = f'；复用 {reused} 帧次' if reused else ''
            self.status.setText(f'预览完成，用时 {result["seconds"]:.1f} 秒{cache_note}；第 {a+1}–{b+1} 帧，{count} 帧待检查，{missing} 帧未检出。尚未写入标注。')
        except Exception as error: self.fail(str(error))

    def refresh_reviews(self):
        self.review_list.clear()
        self.review_range = None
        groups = []
        for i in range(self.start.value()-1, self.end.value()):
            reasons = self.predictions.get(i, {}).get('reasons', [])
            if reasons:
                if groups and groups[-1][1] == i-1 and groups[-1][2] == reasons:
                    groups[-1][1] = i
                else:
                    groups.append([i, i, reasons])
        for start, end, reasons in groups:
            label = str(start+1) if start == end else f'{start+1}–{end+1}'
            item = QtWidgets.QListWidgetItem(f'第 {label} 帧 · '+'；'.join(reasons))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, start)
            item.setData(QtCore.Qt.ItemDataRole.UserRole+1, end)
            item.setToolTip('点击跳到此段开头；可循环播放本段检查。')
            self.review_list.addItem(item)

    def select_review(self, item):
        self.pause()
        self.review_range = (item.data(QtCore.Qt.ItemDataRole.UserRole), item.data(QtCore.Qt.ItemDataRole.UserRole+1))
        self.slider.setValue(self.review_range[0])

    def jump_review(self, direction):
        keys = [self.review_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.review_list.count())]
        if not keys: return
        current = self.slider.value()
        target = next((i for i in (keys if direction > 0 else reversed(keys)) if (i-current)*direction > 0), keys[0] if direction > 0 else keys[-1])
        item = next(self.review_list.item(i) for i in range(self.review_list.count())
                    if self.review_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == target)
        self.review_list.setCurrentItem(item); self.select_review(item)

    def fail(self, message):
        self.recovery_baseline = None
        self.set_busy(False); self.apply.setEnabled(False); self.progress.setRange(0, 100)
        self.status.setText('未完成：'+message)

    def worker_finished(self, *_):
        self.worker_ready = False
        if self.busy: self.fail('AI 进程已退出；标注未改动。日志在本次任务目录中。')

    def cancel_job(self):
        self.recovery_baseline = None
        self.recompute_timer.stop(); self.set_busy(False); self.worker_ready = False
        if self.process is not None:
            self.process.kill(); self.process.waitForFinished(3000)
        self.apply.setEnabled(False); self.progress.setRange(0, 100)
        self.status.setText('已取消，标注未改动；人工关键帧已保留。')

    def toggle_play(self):
        if self.timer.isActive(): self.pause()
        else: self.timer.start(round(1000/self.fps.value())); self.play.setText('暂停')

    def advance(self):
        lo, hi = self.slider.minimum(), self.slider.maximum()
        if self.loop_review.isChecked() and self.review_range and self.review_range[0] <= self.slider.value() <= self.review_range[1]:
            lo, hi = self.review_range
        self.slider.setValue(lo if self.slider.value() >= hi else self.slider.value()+1)

    def step_frame(self, delta):
        self.pause(); self.slider.setValue(self.slider.value()+delta)

    def apply_changes(self):
        self.pause(); self.recompute_timer.stop()
        try:
            if self.busy or self.pending_spans: raise ValueError('请先完成重算，再应用预览。')
            a, b = self.start.value()-1, self.end.value()-1
            predictions = {i: v for i, v in self.predictions.items() if a <= i <= b}
            backup, count = tracking.apply_draft(self.rows, predictions, self.anchors, self.identity)
            self.applied = True; self.apply.setEnabled(False); self.undo.setEnabled(True)
            self.rows = tracking.load_sequence(self.images, self.label_paths, self.identity)
            self.save_anchors()
            self.show_frame(); self.status.setText(f'已更新 {count} 帧；原标注已备份，可撤销本次 AI 补齐。')
        except Exception as error: self.status.setText(str(error))

    def undo_last(self):
        if self.busy: return
        self.pause(); self.recompute_timer.stop()
        try:
            root = self.rows[0]['path'].parent
            backup = tracking.latest_tracking_backup(root)
            if backup is None: raise ValueError('没有可撤销的 AI 补齐。')
            count = tracking.undo_tracking(backup, root)
            self.applied = True
            # Rebase the persisted draft to the restored annotation bytes. An
            # explicit undo does not discard the user's pending human edits.
            self.rows = tracking.load_sequence(self.images, self.label_paths, self.identity)
            self.save_anchors()
            self.load_target(); self.status.setText(f'已恢复 {count} 帧；新建的 JSON 已撤除，原文件恢复。人工关键帧仍可继续使用。')
        except Exception as error: self.status.setText(str(error))

    def done(self, result):
        self.pause(); self.recompute_timer.stop()
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.busy = False; self.process.kill(); self.process.waitForFinished(3000)
        super().done(result)

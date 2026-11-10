"""A native review/apply workflow for a single annotated sequence."""
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from xal_temporal.smoothing import build_preview, apply_preview, latest_backup, undo_preview, review_flags

EDGE_NAMES = ('左', '上', '右', '下')


class BoxComparison(QtWidgets.QWidget):
    edgeEdited = QtCore.pyqtSignal(int, float)
    boxEdited = QtCore.pyqtSignal(object)
    editStarted = QtCore.pyqtSignal()
    pointAdded = QtCore.pyqtSignal(float, float, int)
    maskStroke = QtCore.pyqtSignal(float, float, float, float, int)
    maskStrokeFinished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 340)
        self.row = None
        self.crop = None
        self.image = QtGui.QImage()
        self.follow = True
        self.mapping = None
        self.dragging = None
        self.drag_box = None
        self.allow_move = False
        self.point_mode = None
        self.mask_mode = None
        self.mask_drag = None
        self.setMouseTracking(True)

    def show_row(self, row, crop, follow):
        self.row, self.crop, self.follow = row, crop, follow
        self.drag_box = None
        self.mapping = None
        self.image = QtGui.QImage(str(row['image']))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if not self.row:
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, '设置区间后点击「生成预览」\n左侧原框，右侧平滑框')
            return
        if self.image.isNull():
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, '无法读取图片，请检查原图路径。')
            return
        old = self.row['box']
        new = self.drag_box if self.drag_box is not None else self.row['new_box']
        if self.follow:
            preview_box = self.row['new_box']
            lo = np.minimum(old[:2], preview_box[:2]) - 20
            hi = np.maximum(old[2:], preview_box[2:]) + 20
            if self.row.get('coarse_candidate') is not None:
                alternative = np.asarray(self.row['coarse_candidate'],float)
                lo = np.minimum(lo, alternative[:2]-20)
                hi = np.maximum(hi, alternative[2:]+20)
            crop = QtCore.QRectF(lo[0], lo[1], hi[0]-lo[0], hi[1]-lo[1])
        else:
            crop = self.crop
        half_width = self.width() / 2
        for panel, box, color, title in [(0, old, '#ef6d6d', self.row.get('left_title', '原框')), (1, new, '#25c6c2', self.row.get('right_title', '修正 / 平滑框 · 拖动四边'))]:
            area = QtCore.QRectF(panel*half_width+12, 42, half_width-24, self.height()-58)
            scale = min(area.width()/crop.width(), area.height()/crop.height())
            dest = QtCore.QRectF(area.center().x()-crop.width()*scale/2, area.center().y()-crop.height()*scale/2, crop.width()*scale, crop.height()*scale)
            painter.drawImage(dest, self.image, crop)
            if panel == 0 and self.row.get('baseline_overlay'):
                baseline = self.row['baseline_overlay']
                if not baseline.isNull(): painter.drawImage(dest, baseline, crop)
            if panel == 1 and self.row.get('inspection_overlay'):
                overlay = self.row['inspection_overlay']
                if not isinstance(overlay, QtGui.QImage): overlay = QtGui.QImage(str(overlay))
                if not overlay.isNull(): painter.drawImage(dest, overlay, crop)
            if panel == 1 and self.row.get('coarse_candidate') is not None:
                candidate = self.row['coarse_candidate']
                pen = QtGui.QPen(QtGui.QColor('#ffab35'), 2)
                pen.setStyle(QtCore.Qt.PenStyle.DashLine); painter.setPen(pen)
                candidate_rect = QtCore.QRectF(dest.x()+(candidate[0]-crop.x())*scale,
                    dest.y()+(candidate[1]-crop.y())*scale,
                    (candidate[2]-candidate[0])*scale, (candidate[3]-candidate[1])*scale)
                painter.drawRect(candidate_rect)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.5))
            rect = QtCore.QRectF(dest.x()+(box[0]-crop.x())*scale, dest.y()+(box[1]-crop.y())*scale, (box[2]-box[0])*scale, (box[3]-box[1])*scale)
            visible = self.row.get('old_visible' if panel == 0 else 'new_visible', True)
            if visible:
                painter.drawRect(rect)
            if panel == 1 and visible:
                self.mapping = (dest, crop, scale, rect)
                mids = [(rect.left(), rect.center().y()), (rect.center().x(), rect.top()),
                        (rect.right(), rect.center().y()), (rect.center().x(), rect.bottom())]
                for edge, (x, y) in enumerate(mids):
                    fixed = self.row['fixed_edges'][edge]
                    painter.setPen(QtGui.QPen(QtGui.QColor('#f2c568' if fixed else color), 1.5))
                    painter.setBrush(QtGui.QColor('#f2c568' if fixed else color))
                    painter.drawRect(QtCore.QRectF(x-3,y-3,6,6))
            if panel == 1:
                for x, y, label in self.row.get('prompt_points', []):
                    px, py = dest.x()+(x-crop.x())*scale, dest.y()+(y-crop.y())*scale
                    if dest.contains(QtCore.QPointF(px,py)):
                        painter.setPen(QtGui.QPen(QtGui.QColor('black'), 1))
                        painter.setBrush(QtGui.QColor('#ffe633' if label else '#468bff'))
                        painter.drawEllipse(QtCore.QPointF(px,py), 5, 5)
            painter.drawText(QtCore.QRectF(panel*half_width+12, 8, half_width-24, 26), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, title)

    def hit_edge(self, point):
        if not self.row or self.mapping is None or point.x() < self.width()/2:
            return None
        rect = self.mapping[3]
        if not rect.adjusted(-9,-9,9,9).contains(point):
            return None
        distances = [abs(point.x()-rect.left()),abs(point.y()-rect.top()),
                     abs(point.x()-rect.right()),abs(point.y()-rect.bottom())]
        edge = int(np.argmin(distances))
        return edge if distances[edge] <= 9 else None

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.mask_mode is not None and self.mapping is not None:
            dest, crop, scale, _ = self.mapping
            pos = event.position()
            if pos.x() >= self.width()/2 and dest.contains(pos):
                x,y=crop.x()+(pos.x()-dest.x())/scale,crop.y()+(pos.y()-dest.y())/scale
                self.mask_drag=(x,y)
                self.maskStroke.emit(x,y,x,y,self.mask_mode)
            return
        if self.point_mode is not None and self.mapping is not None:
            dest, crop, scale, _ = self.mapping
            pos = event.position()
            if pos.x() >= self.width()/2 and dest.contains(pos):
                x, y = crop.x()+(pos.x()-dest.x())/scale, crop.y()+(pos.y()-dest.y())/scale
                if 0 <= x < self.image.width() and 0 <= y < self.image.height():
                    self.pointAdded.emit(float(x),float(y),self.point_mode)
            return
        edge = self.hit_edge(event.position())
        if (edge is None and self.allow_move and self.mapping is not None
                and event.position().x() >= self.width()/2
                and self.mapping[3].contains(event.position())):
            self.editStarted.emit()
            self.drag_box = self.row['new_box'].copy()
            self.dragging = ('move', event.position(), self.drag_box.copy(), self.mapping[2])
            return
        if edge is None:
            return
        self.editStarted.emit()
        self.drag_box = self.row['new_box'].copy()
        self.dragging = (edge,event.position(),float(self.drag_box[edge]),self.mapping[2])

    def mouseMoveEvent(self, event):
        if self.mask_drag is not None and self.mapping is not None:
            dest,crop,scale,_=self.mapping;pos=event.position()
            x,y=crop.x()+(pos.x()-dest.x())/scale,crop.y()+(pos.y()-dest.y())/scale
            x0,y0=self.mask_drag;self.maskStroke.emit(x0,y0,x,y,self.mask_mode);self.mask_drag=(x,y)
            return
        if self.dragging:
            edge, origin, initial, scale = self.dragging
            if edge == 'move':
                dx = (event.position().x()-origin.x())/scale
                dy = (event.position().y()-origin.y())/scale
                dx = np.clip(dx, -initial[0], self.row['doc']['imageWidth']-initial[2])
                dy = np.clip(dy, -initial[1], self.row['doc']['imageHeight']-initial[3])
                self.drag_box = initial+np.array([dx,dy,dx,dy])
                self.update()
                return
            delta = event.position().x()-origin.x() if edge in (0,2) else event.position().y()-origin.y()
            value = initial+delta/scale
            # Check against the manually corrected baseline, not a neighboring
            # automatically moved edge that has not been explicitly confirmed.
            base = self.row['baseline_box']
            low = base[edge-2]+.01 if edge in (2,3) else 0
            high = base[edge+2]-.01 if edge in (0,1) else self.row['doc']['imageWidth' if edge==2 else 'imageHeight']
            self.drag_box[edge] = round(float(np.clip(value,low,high)),2)
            self.update()
        else:
            edge = self.hit_edge(event.position())
            cursor = QtCore.Qt.CursorShape.ArrowCursor if edge is None else (
                QtCore.Qt.CursorShape.SizeHorCursor if edge in (0,2) else QtCore.Qt.CursorShape.SizeVerCursor)
            if (edge is None and self.allow_move and self.mapping is not None
                    and event.position().x() >= self.width()/2
                    and self.mapping[3].contains(event.position())):
                cursor = QtCore.Qt.CursorShape.SizeAllCursor
            self.setCursor(cursor)

    def mouseReleaseEvent(self, event):
        if self.mask_drag is not None and event.button()==QtCore.Qt.MouseButton.LeftButton:
            self.mask_drag=None;self.maskStrokeFinished.emit();return
        if self.dragging and event.button()==QtCore.Qt.MouseButton.LeftButton:
            edge = self.dragging[0]
            if edge == 'move':
                value = self.drag_box.tolist()
                self.dragging = None
                self.drag_box = None
                self.boxEdited.emit(value)
                return
            value = float(self.drag_box[edge])
            self.dragging = None
            self.drag_box = None
            self.edgeEdited.emit(edge,value)


class TemporalSmoothingDialog(QtWidgets.QDialog):
    def __init__(self, parent, images, label_paths, identities, current_index=0, selected_identity=None):
        super().__init__(parent)
        self.setWindowTitle('区间平滑 · 人工框辅助')
        self.resize(1180, 850)
        self.images, self.label_paths = images, label_paths
        self.rows = None
        self.applied = False
        self.locks = set()
        self.drafts = {}
        self.display_values = [None]*4
        self.numeric_context = {}
        self.root = Path(label_paths[0]).resolve().parent
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        heading = QtWidgets.QLabel('让人工框连续、平稳地变化')
        f = heading.font(); f.setPointSize(f.pointSize()+3); f.setBold(True); heading.setFont(f)
        layout.addWidget(heading)
        description = QtWidgets.QLabel('右侧拖边修框，修正的边自动固定。只重算相邻锁定帧之间的区间；点击应用才写入标注。')
        description.setWordWrap(True); layout.addWidget(description)
        layout.addSpacing(10)
        body = QtWidgets.QHBoxLayout()
        controls = QtWidgets.QWidget(); controls.setFixedWidth(310)
        form = QtWidgets.QFormLayout(controls); form.setContentsMargins(0,0,16,0); form.setVerticalSpacing(12)
        self.target = QtWidgets.QComboBox()
        for identity in identities:
            self.target.addItem(f'ID {identity[1]} · {identity[0]}', identity)
        if selected_identity in identities:
            self.target.setCurrentIndex(identities.index(selected_identity))
        self.target.setToolTip('按类别和组 ID 精确匹配；重复 ID 或跨接缝双框需要分段处理。')
        form.addRow('目标', self.target)
        self.start = QtWidgets.QSpinBox(); self.end = QtWidgets.QSpinBox()
        for spin in (self.start,self.end): spin.setRange(1,len(images))
        self.start.setValue(current_index+1); self.end.setValue(min(len(images),current_index+30))
        form.addRow('起始帧',self.start); form.addRow('结束帧',self.end)
        self.method=QtWidgets.QComboBox()
        self.method.addItem('已有框去抖','smooth'); self.method.addItem('锁定帧之间线性插值','interpolate')
        form.addRow('处理方式',self.method)
        self.mode=QtWidgets.QComboBox()
        for title,value in [('位置与尺寸','both'),('只平滑位置','position'),('只平滑尺寸','size')]:self.mode.addItem(title,value)
        form.addRow('平滑内容',self.mode)
        edge_controls = QtWidgets.QWidget();edge_layout = QtWidgets.QHBoxLayout(edge_controls);edge_layout.setContentsMargins(0,0,0,0)
        self.edge_enabled = []
        for name in EDGE_NAMES:
            checkbox=QtWidgets.QCheckBox(name);checkbox.setChecked(True)
            checkbox.setToolTip('取消后，此边保留原坐标或你的人工修正。位置/尺寸单独模式使用成对边界。')
            edge_layout.addWidget(checkbox);self.edge_enabled.append(checkbox)
        form.addRow('参与平滑的边',edge_controls)
        self.strength=QtWidgets.QDoubleSpinBox();self.strength.setRange(0,100);self.strength.setValue(4);self.strength.setSingleStep(1)
        form.addRow('平滑强度',self.strength)
        self.limit=QtWidgets.QDoubleSpinBox();self.limit.setRange(0,50);self.limit.setValue(1);self.limit.setDecimals(2);self.limit.setSuffix(' px')
        self.limit.setToolTip('限制自动平滑相对原框的改动；人工拖边不受此限制，确认后作为新的锁定约束。')
        form.addRow('最大边界改动',self.limit)
        self.locked=QtWidgets.QLineEdit();self.locked.setPlaceholderText('例如 8, 15, 22')
        self.locked.setToolTip('输入列表中的帧序号，使用逗号或空格分隔；这些帧的原框保持不变。')
        form.addRow('锁定中间帧',self.locked)
        self.generate=QtWidgets.QPushButton('生成预览'); self.generate.clicked.connect(self.generate_preview)
        form.addRow(self.generate)
        self.follow=QtWidgets.QCheckBox('跟随目标放大');self.follow.setChecked(True)
        self.follow.setToolTip('左右使用相同裁剪；取消后固定视野，查看真实运动轨迹。')
        form.addRow(self.follow)
        self.warning=QtWidgets.QLabel('拖右侧框边或输入坐标即可修正。\n金色手柄表示固定的边。人工修正不受自动改动上限限制。')
        self.warning.setWordWrap(True);form.addRow(self.warning)
        body.addWidget(controls)
        preview_layout=QtWidgets.QVBoxLayout()
        self.comparison=BoxComparison();preview_layout.addWidget(self.comparison,1)
        self.frame_info=QtWidgets.QLabel('尚未生成预览');self.frame_info.setWordWrap(True);preview_layout.addWidget(self.frame_info)
        self.slider=QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal);self.slider.setEnabled(False);preview_layout.addWidget(self.slider)
        editors=QtWidgets.QGridLayout();self.edge_values=[];self.frame_edge_locks=[]
        for edge,name in enumerate(EDGE_NAMES):
            label=QtWidgets.QLabel(name+'边坐标')
            spin=QtWidgets.QDoubleSpinBox();spin.setDecimals(2);spin.setSingleStep(.25);spin.setEnabled(False)
            spin.setAccessibleName(name+'边坐标');spin.setKeyboardTracking(False)
            lock=QtWidgets.QCheckBox('固定'+name+'边');lock.setEnabled(False)
            lock.setToolTip('固定当前预览坐标；取消固定会清除此边的预览修正。首尾帧和整帧关键帧不能在这里解除。')
            spin.editingFinished.connect(lambda e=edge:self.finish_numeric_edit(e))
            spin.installEventFilter(self);spin.lineEdit().installEventFilter(self)
            lock.toggled.connect(lambda checked,e=edge:self.set_edge_lock(e,checked))
            editors.addWidget(label,0,edge);editors.addWidget(spin,1,edge);editors.addWidget(lock,2,edge)
            self.edge_values.append(spin);self.frame_edge_locks.append(lock)
        preview_layout.addLayout(editors)
        playback=QtWidgets.QHBoxLayout()
        self.play=QtWidgets.QPushButton('播放');self.play.setEnabled(False);self.play.clicked.connect(self.toggle_play)
        self.lock_current=QtWidgets.QPushButton('锁定当前帧');self.lock_current.setEnabled(False);self.lock_current.clicked.connect(self.lock_frame)
        self.clear_current=QtWidgets.QPushButton('清除此帧修正');self.clear_current.setEnabled(False);self.clear_current.clicked.connect(self.clear_frame)
        self.fps=QtWidgets.QSpinBox();self.fps.setRange(1,30);self.fps.setValue(8);self.fps.setSuffix(' 帧/秒');self.fps.setFixedWidth(105)
        playback.addWidget(self.play);playback.addWidget(self.lock_current);playback.addWidget(self.clear_current);playback.addStretch();playback.addWidget(self.fps)
        preview_layout.addLayout(playback)
        preview_layout.addWidget(QtWidgets.QLabel('待检查帧 · 提示不代表标注错误'))
        self.review_list=QtWidgets.QListWidget();self.review_list.setFixedHeight(65)
        self.review_list.itemClicked.connect(lambda item:self.slider.setValue(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        preview_layout.addWidget(self.review_list)
        body.addLayout(preview_layout,1);layout.addLayout(body,1)
        self.status=QtWidgets.QLabel('当前标注目录：'+str(self.root));self.status.setWordWrap(True);layout.addWidget(self.status)
        buttons=QtWidgets.QHBoxLayout()
        self.undo=QtWidgets.QPushButton('撤销上次区间平滑');self.undo.clicked.connect(self.undo_last)
        self.undo.setEnabled(latest_backup(self.root) is not None)
        buttons.addWidget(self.undo);buttons.addStretch()
        close=QtWidgets.QPushButton('关闭');close.clicked.connect(self.reject);buttons.addWidget(close)
        self.apply=QtWidgets.QPushButton('应用到所选帧');self.apply.setEnabled(False);self.apply.clicked.connect(self.apply_changes);buttons.addWidget(self.apply)
        layout.addLayout(buttons)
        self.slider.valueChanged.connect(self.show_frame);self.follow.toggled.connect(self.show_frame)
        for control in (self.target,self.method,self.mode):control.currentIndexChanged.connect(self.invalidate)
        for control in (self.start,self.end,self.strength,self.limit):control.valueChanged.connect(self.invalidate)
        self.locked.textChanged.connect(self.invalidate)
        for checkbox in self.edge_enabled:checkbox.toggled.connect(self.invalidate)
        self.comparison.editStarted.connect(self.pause)
        self.comparison.edgeEdited.connect(self.edit_edge)
        self.fps.valueChanged.connect(lambda value:self.timer.setInterval(round(1000/value)))
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False);button.setDefault(False)

    def pause(self):
        self.timer.stop();self.play.setText('播放')

    def eventFilter(self, obj, event):
        if event.type()==QtCore.QEvent.Type.FocusIn:
            for edge,spin in enumerate(getattr(self,'edge_values',[])):
                if obj in (spin,spin.lineEdit()):
                    self.pause()
                    if self.rows:
                        self.numeric_context[edge]=(self.target.currentData(),self.start.value()+self.slider.value())
                    break
        return super().eventFilter(obj,event)

    def finish_numeric_edit(self, edge):
        if not self.rows:return
        context=(self.target.currentData(),self.start.value()+self.slider.value())
        if self.numeric_context.get(edge,context)!=context:
            self.status.setText('帧已切换，坐标输入未应用，请重新输入。')
            self.show_frame();return
        value=self.edge_values[edge].value()
        # Focus/blur must not pin a rounded display of a subpixel result.
        if self.display_values[edge] is None or abs(value-self.display_values[edge])<1e-9:return
        self.edit_edge(edge,value)

    def draft(self):
        return self.drafts.setdefault(self.target.currentData(),{'edits':{},'locks':{}})

    def invalidate(self, *_):
        self.pause();self.rows=None
        self.apply.setEnabled(False);self.play.setEnabled(False);self.slider.setEnabled(False);self.lock_current.setEnabled(False)
        self.comparison.row=None;self.comparison.update()
        self.frame_info.setText('设置已改变，请重新生成预览。')
        self.strength.setEnabled(self.method.currentData()=='smooth')
        for checkbox in self.edge_enabled:checkbox.setEnabled(self.mode.currentData()=='both')
        for control in self.edge_values+self.frame_edge_locks+[self.clear_current]:control.setEnabled(False)
        self.review_list.clear()

    def generate_preview(self):
        current=self.slider.value()
        self.invalidate()
        try:
            start,end=self.start.value(),self.end.value()
            if end-start<2:raise ValueError('结束帧至少比起始帧大 2。')
            text=self.locked.text().replace('，',',').replace(',', ' ')
            try:
                locks={int(x) for x in text.split()}
            except ValueError:
                raise ValueError('锁定帧请输入整数，用逗号或空格分隔。') from None
            if any(i<start or i>end for i in locks):raise ValueError('锁定帧必须在起止区间内。')
            draft=self.draft()
            edits={i-start:values for i,values in draft['edits'].items() if start<=i<=end}
            edge_locks={i-start:edges for i,edges in draft['locks'].items() if start<=i<=end}
            enabled=[c.isChecked() for c in self.edge_enabled] if self.mode.currentData()=='both' else [True]*4
            self.rows=build_preview(self.images[start-1:end],self.label_paths[start-1:end],self.target.currentData(),{i-start for i in locks},manual_overrides=edits,edge_locks=edge_locks,enabled_edges=enabled,strength=self.strength.value(),limit=self.limit.value(),mode=self.mode.currentData(),method=self.method.currentData())
            self.locks=locks|{start,end}
            b=np.array([r['box'] for r in self.rows]);new=np.array([r['new_box'] for r in self.rows])
            lo=np.minimum(b[:,:2].min(axis=0),new[:,:2].min(axis=0))-20
            hi=np.maximum(b[:,2:].max(axis=0),new[:,2:].max(axis=0))+20
            self.crop=QtCore.QRectF(lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1])
            self.slider.setRange(0,len(self.rows)-1);self.slider.setValue(min(current,len(self.rows)-1));self.slider.setEnabled(True)
            for c in (self.apply,self.play,self.lock_current,self.clear_current):c.setEnabled(True)
            baseline=np.array([r['baseline_box'] for r in self.rows])
            change=float(np.max(abs(new-baseline)))
            manual_count=sum(int(np.sum(r['manual_edges'])) for r in self.rows)
            flags=review_flags(self.rows,self.limit.value())
            for index,reasons in flags.items():
                item=QtWidgets.QListWidgetItem(f'第 {start+index} 帧 · '+ '；'.join(reasons))
                item.setData(QtCore.Qt.ItemDataRole.UserRole,index);self.review_list.addItem(item)
            self.status.setText(f'预览 {len(self.rows)} 帧；自动改动最多 {change:.2f} px；人工固定 {manual_count} 条边；{len(flags)} 帧待检查。尚未写入标注。')
            self.show_frame()
        except Exception as error:
            self.invalidate();self.status.setText(str(error))

    def show_frame(self, *_):
        if not self.rows:return
        index=self.slider.value();row=self.rows[index];number=self.start.value()+index
        delta=row['new_box']-row['box']
        self.comparison.show_row(row,self.crop,self.follow.isChecked())
        suffix=' · 已锁定' if number in self.locks else ''
        if number not in self.locks and row['fixed_edges'].any():
            suffix=' · 固定 '+ '/'.join(EDGE_NAMES[e] for e in range(4) if row['fixed_edges'][e])+' 边'
        self.frame_info.setText(f'第 {number} 帧 · {row["image"].name}{suffix}\n边界改动（左 / 上 / 右 / 下）：'+ ' / '.join(f'{v:+.2f}' for v in delta)+' px')
        base=row['baseline_box']
        for edge,spin in enumerate(self.edge_values):
            blocker=QtCore.QSignalBlocker(spin)
            low=base[edge-2]+.01 if edge in (2,3) else 0
            high=base[edge+2]-.01 if edge in (0,1) else row['doc']['imageWidth' if edge==2 else 'imageHeight']
            spin.setRange(low,high);spin.setValue(float(row['new_box'][edge]));spin.setEnabled(True)
            self.display_values[edge]=spin.value()
            del blocker
        for edge,checkbox in enumerate(self.frame_edge_locks):
            blocker=QtCore.QSignalBlocker(checkbox)
            checkbox.setChecked(bool(row['fixed_edges'][edge]))
            checkbox.setEnabled(number not in self.locks and self.mode.currentData()=='both' and self.edge_enabled[edge].isChecked())
            del blocker

    def toggle_play(self):
        if self.timer.isActive():self.timer.stop();self.play.setText('播放')
        else:self.timer.start(round(1000/self.fps.value()));self.play.setText('暂停')

    def advance(self):
        if self.rows:self.slider.setValue((self.slider.value()+1)%len(self.rows))

    def lock_frame(self):
        if not self.rows:return
        number=self.start.value()+self.slider.value()
        self.draft()['edits'][number]={edge:float(value) for edge,value in enumerate(self.rows[self.slider.value()]['new_box'])}
        locks=set(self.locks)|{number}
        locks.discard(self.start.value());locks.discard(self.end.value())
        self.locked.setText(', '.join(map(str,sorted(locks))))
        self.generate_preview()
        if self.rows:self.slider.setValue(number-self.start.value())

    def edit_edge(self, edge, value):
        if not self.rows:return
        self.pause()
        index=self.slider.value();number=self.start.value()+index;row=self.rows[index]
        if abs(value-row['new_box'][edge])<1e-7:return
        proposed=row['baseline_box'].copy();proposed[edge]=value
        if (not np.isfinite(value) or np.any(proposed[2:]<=proposed[:2]) or np.any(proposed[:2]<0)
                or proposed[2]>row['doc']['imageWidth'] or proposed[3]>row['doc']['imageHeight']):
            self.status.setText('边界越界或交叉，请调整坐标。');self.show_frame();return
        self.draft()['edits'].setdefault(number,{})[edge]=float(value)
        self.draft()['locks'].setdefault(number,set()).add(edge)
        # One-edge edits need independent edge variables, not coupled center/size.
        if self.mode.currentData()!='both':
            blocker=QtCore.QSignalBlocker(self.mode);self.mode.setCurrentIndex(0);del blocker
        self.generate_preview()

    def set_edge_lock(self, edge, checked):
        if not self.rows:return
        number=self.start.value()+self.slider.value();draft=self.draft()
        if checked:
            value=float(self.rows[self.slider.value()]['new_box'][edge])
            draft['edits'].setdefault(number,{})[edge]=value
            draft['locks'].setdefault(number,set()).add(edge)
        else:
            draft['edits'].get(number,{}).pop(edge,None)
            draft['locks'].get(number,set()).discard(edge)
        self.generate_preview()

    def clear_frame(self):
        if not self.rows:return
        number=self.start.value()+self.slider.value();draft=self.draft()
        draft['edits'].pop(number,None);draft['locks'].pop(number,None)
        locks=set(self.locks)-{number,self.start.value(),self.end.value()}
        blocker=QtCore.QSignalBlocker(self.locked)
        self.locked.setText(', '.join(map(str,sorted(locks))));del blocker
        self.generate_preview()

    def apply_changes(self):
        self.timer.stop();self.play.setText('播放')
        try:
            backup=apply_preview(self.rows)
            self.applied=True;self.undo.setEnabled(True);self.apply.setEnabled(False)
            self.status.setText('已应用。原坐标已自动备份，可撤销：'+str(backup))
        except Exception as error:self.status.setText(str(error))

    def undo_last(self):
        self.timer.stop()
        try:
            backup=latest_backup(self.root)
            if backup is None:raise ValueError('当前文件夹没有可撤销的区间平滑。')
            count=undo_preview(backup,self.root)
            self.drafts.pop(self.target.currentData(),None)
            self.applied=True;self.invalidate();self.undo.setEnabled(latest_backup(self.root) is not None)
            self.status.setText(f'已恢复 {count} 帧的原始标注。')
        except Exception as error:self.status.setText(str(error))

    def done(self, result):
        self.timer.stop()
        super().done(result)

"""Sparse human keyframes, AI drafts and reversible annotation transactions.

This module deliberately has no torch or Qt dependency. Model predictions never
become human keyframes unless the user explicitly confirms or edits them.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from datetime import datetime

import numpy as np
from PIL import Image

from .smoothing import atomic_write, bounds, replace_bounds, sha
from .geometry import box_iou, merge_directions
from .runtime import default_home

# Writable home for sessions, jobs and the optional SAM2 config.
# Tests patch this name. It is not the annotation folder and not the install tree.
PROJECT_ROOT = default_home()


def read_optional(path):
    path = Path(path)
    return path.read_bytes() if path.exists() else None


def valid_box(box, size):
    box = np.asarray(box, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or np.any(box[:2] < 0) or np.any(box[2:] > size)
            or np.any(box[2:] <= box[:2])):
        raise ValueError('矩形越界或宽高无效。')
    return box.tolist()


def visible_source_box(box, size):
    """Clip a source annotation only for preview; never modify the source JSON."""
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None, '原框坐标无效：保留原标注，需人工检查'
    visible = np.clip(box, 0, np.tile(size, 2))
    if np.any(visible[2:] <= visible[:2]):
        return None, '原框完全位于画面外：保留原标注，需人工检查'
    if not np.array_equal(visible, box):
        return visible.tolist(), '原框超出画面：预览只显示可见区域，保留原标注'
    return box.tolist(), None


def load_sequence(images, label_paths, identity):
    if not images or len(images) != len(label_paths) or identity[1] is None:
        raise ValueError('请选择带组 ID 的矩形目标和有效视频帧。')
    images = [Path(p).resolve() for p in images]
    paths = [Path(p).resolve() for p in label_paths]
    if (len({p.parent for p in paths}) != 1 or len({p.parent for p in images}) != 1
            or len(set(paths)) != len(paths) or len(set(images)) != len(images)):
        raise ValueError('一次只能处理一个序列；图片或标注路径不能重复。')
    if all(p.stem.isdigit() for p in images):
        numbers = [int(p.stem) for p in images]
        if any(b-a != 1 for a, b in zip(numbers, numbers[1:])):
            raise ValueError('帧编号不连续，请分段处理。')
    rows = []
    for image, path in zip(images, paths):
        with Image.open(image) as im:
            size = im.size
        raw = read_optional(path)
        if raw is None:
            doc = {'version': '4.0.0-beta.7', 'flags': {}, 'shapes': [],
                   'imagePath': os.path.relpath(image, path.parent), 'imageData': None,
                   'imageWidth': size[0], 'imageHeight': size[1], 'checked': False}
        else:
            doc = json.loads(raw.decode('utf-8-sig'))
            if not isinstance(doc.get('shapes'), list):
                raise ValueError(f'{path.name} 标注结构无效。')
            if (doc.get('imageWidth'), doc.get('imageHeight')) != size:
                raise ValueError(f'{path.name} 的标注尺寸与图片不一致。')
        matches = [(i, s) for i, s in enumerate(doc['shapes'])
                   if (s.get('label'), s.get('group_id')) == tuple(identity)]
        if len(matches) > 1:
            raise ValueError(f'{path.name} 同一目标有多个框，跨接缝目标请分段处理。')
        index, shape = matches[0] if matches else (None, None)
        if shape is not None and shape.get('shape_type') != 'rectangle':
            raise ValueError(f'{path.name} 的同 ID 目标不是水平矩形。')
        source_box = bounds(shape) if shape else None
        box, source_box_issue = visible_source_box(source_box, size) if shape else (None, None)
        stat = image.stat()
        rows.append({'image': image, 'path': path, 'raw': raw, 'doc': doc,
                     'index': index, 'box': box, 'source_box': source_box,
                     'source_box_issue': source_box_issue, 'size': size,
                     'image_stamp': [stat.st_size, stat.st_mtime_ns]})
    if len({r['size'] for r in rows}) != 1:
        raise ValueError('帧尺寸不一致，请分段处理。')
    return rows


def legacy_session_path(rows, identity):
    key = hashlib.sha256(json.dumps(list(identity), ensure_ascii=False).encode()).hexdigest()[:20]
    return rows[0]['path'].parent / '.temporal_tracking' / (key + '.json')


def session_path(rows, identity):
    # New sessions are isolated from the source sequence and its old cache.
    key = hashlib.sha256(json.dumps([str(rows[0]['path'].parent.resolve()), list(identity)], ensure_ascii=False).encode()).hexdigest()
    return PROJECT_ROOT/'runtime'/'tracking_sessions'/(key+'.json')


def readable_session_path(rows, identity):
    path = session_path(rows, identity)
    return path if path.exists() else legacy_session_path(rows, identity)


def readable_draft_path(rows, identity):
    path = session_path(rows, identity).with_suffix('.draft.json')
    return path if path.exists() else legacy_session_path(rows, identity).with_suffix('.draft.json')


def tracking_options(options=None):
    mode = (options or {}).get('boundary_mode', 'mask')
    if mode not in ('mask', 'rotor'): raise ValueError('未知的框边界模式。')
    return {'boundary_mode': mode, 'boundary_revision': 5 if mode == 'rotor' else 0}


def saved_tracking_options(rows, identity):
    path = readable_session_path(rows, identity)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if 'options' in data: return tracking_options(data['options'])
    label = str(identity[0]).lower()
    rotor = any(word in label for word in ('drone', 'quadcopter', 'uav', '无人机', '旋翼'))
    return tracking_options({'boundary_mode': 'rotor' if rotor else 'mask'})


def save_keyframes(rows, identity, anchors, options=None):
    path = session_path(rows, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'version': 1, 'identity': list(identity), 'options': tracking_options(options), 'anchors': [
        {'image': rows[i]['image'].name, 'stamp': rows[i]['image_stamp'],
         'annotation_sha': sha(rows[i]['raw']) if rows[i]['raw'] is not None else None,
         'box': valid_box(box, rows[i]['size'])} for i, box in sorted(anchors.items())]}
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))


def load_keyframes(rows, identity):
    path = readable_session_path(rows, identity)
    if not path.exists():
        return {}, []
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('identity') != list(identity):
        raise ValueError('人工关键帧记录不匹配。')
    by_name = {r['image'].name: i for i, r in enumerate(rows)}
    anchors, stale = {}, []
    for entry in data['anchors']:
        i = by_name.get(entry['image'])
        if (i is None or rows[i]['image_stamp'] != entry.get('stamp') or
                entry.get('annotation_sha') != (sha(rows[i]['raw']) if rows[i]['raw'] is not None else None)):
            stale.append(entry['image'])
        else:
            anchors[i] = valid_box(entry['box'], rows[i]['size'])
    return anchors, stale


def tracking_intervals(anchors, start, end):
    keys = sorted(i for i in anchors if start <= i <= end)
    if not keys:
        raise ValueError('处理区间内至少需要一个人工关键帧。')
    knots = sorted(set([start, end] + keys))
    return [(a, b) for a, b in zip(knots, knots[1:])]


def draft_signature(rows):
    return sha(json.dumps([[r['image'].name, r['image_stamp'],
                           sha(r['raw']) if r['raw'] is not None else None]
                          for r in rows], ensure_ascii=False).encode())



def prompt_path(rows, identity):
    # Keep experimental prompts outside the source sequence and its existing caches.
    key = hashlib.sha256(json.dumps([str(rows[0]['path'].parent.resolve()), list(identity)], ensure_ascii=False).encode()).hexdigest()
    return PROJECT_ROOT/'runtime'/'tracking_prompts'/(key+'.json')


def save_prompts(rows, identity, anchors, prompts):
    path = prompt_path(rows, identity)
    entries = []
    for i, points in sorted(prompts.items()):
        if i not in anchors or not points: continue
        clean = []
        for point in points:
            if len(point) != 3: raise ValueError('部件提示点格式无效。')
            x,y,label = point
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < rows[i]['size'][0] and 0 <= y < rows[i]['size'][1] and label in (0,1)):
                raise ValueError('部件提示点越界或类型无效。')
            clean.append([float(x),float(y),int(label)])
        entries.append({'image':rows[i]['image'].name,'stamp':rows[i]['image_stamp'], 'box':list(anchors[i]),'points':clean})
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(path, json.dumps({'version':1,'identity':list(identity),'entries':entries},ensure_ascii=False).encode('utf8'))


def load_prompts(rows, identity, anchors):
    path = prompt_path(rows, identity)
    if not path.exists(): return {}
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('version') != 1 or data.get('identity') != list(identity): return {}
    names = {row['image'].name:i for i,row in enumerate(rows)}
    result = {}
    for entry in data.get('entries',[]):
        i=names.get(entry.get('image'))
        if i is None or i not in anchors or entry.get('stamp') != rows[i]['image_stamp'] or entry.get('box') != list(anchors[i]): continue
        points=entry.get('points',[])
        if all(isinstance(p,list) and len(p)==3 and all(isinstance(v,(int,float)) and np.isfinite(v) for v in p) and 0 <= p[0] < rows[i]['size'][0] and 0 <= p[1] < rows[i]['size'][1] and p[2] in (0,1) for p in points):
            if points: result[i]=points
    return result


def prompt_draft_path(rows, identity, prompts, masks=None):
    if masks: return prompt_path(rows,identity).with_suffix('.mask.draft.json')
    if not prompts: return session_path(rows,identity).with_suffix('.draft.json')
    return prompt_path(rows,identity).with_suffix('.draft.json')

def save_draft(rows, identity, anchors, predictions, frame_range, options=None, prompts=None, masks=None):
    path = prompt_draft_path(rows, identity, prompts, masks)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'version': 1, 'identity': list(identity), 'signature': draft_signature(rows),
            'anchors': {str(i): b for i, b in anchors.items()}, 'range': list(frame_range),
            'predictions': predictions, 'options': tracking_options(options)}
    if prompts: data['prompts'] = {str(i): points for i,points in sorted(prompts.items())}
    if masks: data['masks'] = {str(i): v['sha256'] for i,v in sorted(masks.items())}
    atomic_write(path, json.dumps(data, ensure_ascii=False).encode('utf-8'))


def load_draft(rows, identity, anchors, options=None, prompts=None, masks=None):
    path = prompt_draft_path(rows, identity, prompts, masks) if prompts or masks else readable_draft_path(rows, identity)
    if not path.exists(): return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data.get('version') != 1 or data.get('identity') != list(identity)
            or data.get('signature') != draft_signature(rows)
            or data.get('anchors') != {str(i): b for i, b in anchors.items()}
            or data.get('options', tracking_options()) != tracking_options(options)
            or data.get('prompts', {}) != {str(i): points for i,points in sorted((prompts or {}).items())}
            or data.get('masks', {}) != {str(i): v['sha256'] for i,v in sorted((masks or {}).items())}):
        return None
    start, end = data['range']
    if not 0 <= start <= end < len(rows): return None
    predictions = {int(i): v for i, v in data['predictions'].items()}
    if not all(i in predictions for i in range(start, end+1)): return None
    for i, item in predictions.items():
        if not 0 <= i < len(rows): return None
        if item.get('box') is not None: valid_box(item['box'], rows[i]['size'])
    return predictions, (start, end)


def saved_draft_range(rows, identity, anchors):
    """Migrate a valid old scope without reusing old boundary predictions."""
    path = readable_draft_path(rows, identity)
    if not path.exists(): return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data.get('identity') != list(identity) or data.get('signature') != draft_signature(rows)
            or data.get('anchors') != {str(i): b for i, b in anchors.items()}):
        return None
    scope = data.get('range', [])
    if len(scope) == 2 and all(isinstance(i, int) for i in scope) and 0 <= scope[0] <= scope[1] < len(rows):
        return tuple(scope)
    return None


def affected_span(anchors, frame, start, end):
    others = sorted(i for i in anchors if start <= i <= end and i != frame)
    return (max([start] + [i for i in others if i < frame]),
            min([end] + [i for i in others if i > frame]))


def review_predictions(predictions, anchors, rows):
    result = copy.deepcopy(predictions)
    previous = None
    for i in sorted(result):
        item = result[i]
        item['reasons'] = [r for r in item.get('reasons', []) if r not in
                           ('目标接近画面边缘', '框尺寸突变', '框位置跳变', '与已有框差异较大', 'AI 框内缩，请核对旋翼浅灰边')]
        if i in anchors:
            item['box'], item['reasons'] = list(anchors[i]), list(item.get('edge_review_reasons',[]))
        if rows[i].get('source_box_issue') and rows[i]['source_box_issue'] not in item['reasons']:
            item['reasons'].append(rows[i]['source_box_issue'])
        box = item.get('box')
        if box is None:
            previous = None
            continue
        box = np.asarray(valid_box(box, rows[i]['size']))
        if i not in anchors:
            if rows[i]['box'] is not None and box_iou(box, rows[i]['box']) < .6:
                item['reasons'].append('与已有框差异较大')
            if item.get('boundary_mode') == 'rotor' and rows[i]['box'] is not None:
                inner = (box-np.asarray(rows[i]['box']))*np.array([1, 1, -1, -1])
                if np.max(inner) > 1.5 and box_iou(box, rows[i]['box']) >= .6:
                    item['reasons'].append('AI 框内缩，请核对旋翼浅灰边')
            w, h = rows[i]['size']
            if min(box[0], box[1], w-box[2], h-box[3]) <= 1:
                item['reasons'].append('目标接近画面边缘')
            if previous is not None:
                old = previous
                ratio = (box[2:]-box[:2])/(old[2:]-old[:2])
                distance = np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2)
                if np.any((ratio > 1.5) | (ratio < 1/1.5)):
                    item['reasons'].append('框尺寸突变')
                if distance > max(16, .6*np.linalg.norm(old[2:]-old[:2])):
                    item['reasons'].append('框位置跳变')
        item['reasons'] = list(dict.fromkeys(item['reasons']))
        previous = box
    return result


def new_shape(template, box):
    # Carry identity/description, but never clone frame-specific confidence,
    # masks, visibility or another frame's review flags.
    keys = ('label', 'group_id', 'description', 'caption', 'short_caption', 'category')
    shape = {k: copy.deepcopy(template[k]) for k in keys if k in template}
    identity_keys = ('target_id', 'global_track_id', 'category_track_id', 'category',
                     'original_global_track_id', 'short_caption')
    shape.update(shape_type='rectangle', points=[[box[0], box[1]], [box[2], box[1]],
                 [box[2], box[3]], [box[0], box[3]]], flags={}, score=None,
                 difficult=False, kie_linking=[], attributes={
                     k: copy.deepcopy(v) for k, v in template.get('attributes', {}).items()
                     if k in identity_keys})
    return shape


def build_writes(rows, predictions, anchors, identity):
    template = next((r['doc']['shapes'][r['index']] for r in rows if r['index'] is not None), None)
    if template is None:
        raise ValueError('找不到目标的类别和 ID 模板。')
    writes = []
    for i, item in sorted(predictions.items()):
        row = rows[i]
        box = anchors.get(i, item.get('box'))
        # Do not silently replace an out-of-image existing annotation. An
        # explicit human keyframe edit can still deliberately correct it.
        if row.get('source_box_issue') and (i not in anchors or
                (row['box'] is not None and np.allclose(anchors[i],row['box'],rtol=0,atol=1e-8))):
            continue
        if box is None:
            continue
        box = valid_box(box, row['size'])
        if row['box'] is not None and np.max(np.abs(np.asarray(box)-row['box'])) < 1e-8:
            continue
        doc = copy.deepcopy(row['doc'])
        if row['index'] is None:
            shape = new_shape(template, box)
            doc['shapes'].append(shape)
        else:
            shape = doc['shapes'][row['index']]
            replace_bounds(shape, box)
        # A batch assistant may supply an initialization/correction, but that
        # is not a user-confirmed human annotation.
        human_keyframe = i in anchors and item.get('prompt_source') != 'assistant'
        shape.setdefault('flags', {})['temporal_ai_generated'] = not human_keyframe
        shape.setdefault('attributes', {})['temporal_ai_tracking'] = {
            'backend': 'sam2.1-small', 'human_keyframe': human_keyframe,
            'boundary_mode': item.get('boundary_mode', 'mask'),
            'review_reasons': item.get('reasons', [])}
        if item.get('prompt_source') == 'assistant':
            shape['attributes']['temporal_ai_tracking']['prompt_source'] = 'assistant'
        if not human_keyframe:
            shape['score'] = None
        doc['checked'] = False
        raw = (json.dumps(doc, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
        writes.append({'path': row['path'], 'before': row['raw'], 'after': raw})
    return writes


def apply_draft(rows, predictions, anchors, identity):
    if not predictions:
        raise ValueError('请先完成 AI 预览。')
    root = rows[0]['path'].parent.resolve()
    for row in rows:
        if row['path'].resolve().parent != root:
            raise ValueError('禁止跨标注目录写入。')
        if read_optional(row['path']) != row['raw']:
            raise ValueError(f'{row["path"].name} 在预览后被修改，请重新载入。')
        stat = row['image'].stat()
        if [stat.st_size, stat.st_mtime_ns] != row['image_stamp']:
            raise ValueError('图片在预览后发生变化，请重新载入。')
    writes = build_writes(rows, predictions, anchors, identity)
    if not writes:
        raise ValueError('没有需要写入的框。')
    backup = root / '.temporal_tracking_backups' / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    receipt = {'version': 1, 'root': str(root), 'status': 'prepared', 'files': []}
    for w in writes:
        if w['before'] is not None:
            (backup / w['path'].name).write_bytes(w['before'])
        receipt['files'].append({'name': w['path'].name, 'before': sha(w['before']) if w['before'] is not None else None,
                                 'after': sha(w['after'])})
    atomic_write(backup/'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode())
    done = []
    try:
        for w in writes:
            if read_optional(w['path']) != w['before']:
                raise ValueError('标注被其他程序修改，应用已取消。')
            atomic_write(w['path'], w['after'])
            done.append(w)
        receipt['status'] = 'applied'
        atomic_write(backup/'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode())
    except Exception:
        for w in reversed(done):
            if w['before'] is None:
                w['path'].unlink()
            else:
                atomic_write(w['path'], w['before'])
        raise
    return backup, len(writes)


def latest_tracking_backup(root):
    folder = Path(root)/'.temporal_tracking_backups'
    for path in sorted(folder.glob('*/transaction.json'), reverse=True):
        if json.loads(path.read_text(encoding='utf-8')).get('status') == 'applied':
            return path.parent
    return None


def undo_tracking(backup, root):
    backup, root = Path(backup).resolve(), Path(root).resolve()
    if backup.parent != root/'.temporal_tracking_backups':
        raise ValueError('备份不属于当前标注目录。')
    receipt = json.loads((backup/'transaction.json').read_text(encoding='utf-8'))
    if receipt.get('root') != str(root) or receipt.get('status') != 'applied':
        raise ValueError('备份状态无效。')
    writes = []
    for entry in receipt['files']:
        name = entry['name']
        if Path(name).name != name or not name.endswith('.json') or (root/name).resolve().parent != root:
            raise ValueError('备份路径无效。')
        path = root/name
        after = read_optional(path)
        before = (backup/name).read_bytes() if entry['before'] is not None else None
        if after is None or sha(after) != entry['after'] or (before is not None and sha(before) != entry['before']):
            raise ValueError(f'{name} 在应用后被修改，撤销已停止以保护新修改。')
        writes.append((path, before, after))
    done = []
    try:
        for path, before, after in writes:
            if read_optional(path) != after:
                raise ValueError('标注被其他程序修改，撤销已停止。')
            if before is None:
                path.unlink()
            else:
                atomic_write(path, before)
            done.append((path, after))
        receipt['status'] = 'undone'
        atomic_write(backup/'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode())
    except Exception:
        for path, after in reversed(done):
            atomic_write(path, after)
        raise
    return len(writes)


def runtime_config():
    path = PROJECT_ROOT/'runtime/tracking_runtime.json'
    if not path.exists():
        raise ValueError('AI 运行环境尚未安装完成。')
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ('python', 'checkpoint'):
        value = Path(config[key])
        config[key] = str((PROJECT_ROOT/value if not value.is_absolute() else value).resolve())
    if not Path(config['python']).is_file() or not Path(config['checkpoint']).is_file():
        raise ValueError('AI Python 或模型文件不存在，请检查运行环境。')
    return config

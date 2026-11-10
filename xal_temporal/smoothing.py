"""Model-free rectangle smoothing and reversible, conflict-checked writes."""
import copy
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounds(shape):
    pts = np.asarray(shape.get('points'), dtype=float)
    if pts.shape not in ((2, 2), (4, 2)) or not np.isfinite(pts).all():
        raise ValueError('矩形坐标无效：只支持两点或四点矩形。')
    b = np.r_[pts.min(axis=0), pts.max(axis=0)]
    if np.any(b[2:] <= b[:2]):
        raise ValueError('矩形宽高必须大于零。')
    if len(pts) == 4 and any(
        not (x in (b[0], b[2]) and y in (b[1], b[3])) for x, y in pts
    ):
        raise ValueError('仅支持水平矩形，请将旋转框分开处理。')
    return b


def replace_bounds(shape, box):
    before = np.asarray(shape['points'], dtype=float)
    after = before.copy()
    for axis in (0, 1):
        after[:, axis] = np.where(
            before[:, axis] == before[:, axis].min(), box[axis], box[axis + 2]
        )
    shape['points'] = after.tolist()


def smooth_boxes(boxes, anchors, strength=4.0, limit=1.0, mode='both', method='smooth', image_size=None):
    """Bounded position/size smoothing; anchors are exact, never synthesized."""
    b = np.asarray(boxes, dtype=float)
    if b.ndim != 2 or b.shape[1] != 4 or len(b) < 3 or not np.isfinite(b).all():
        raise ValueError('至少选择三帧有效矩形。')
    if np.any(b[:, 2:] <= b[:, :2]) or not np.isfinite([strength, limit]).all() or limit < 0 or strength < 0:
        raise ValueError('宽高、平滑强度或改动范围无效。')
    if mode not in ('both', 'position', 'size') or method not in ('smooth', 'interpolate'):
        raise ValueError('未知的平滑模式。')
    locked = sorted(set(anchors) | {0, len(b) - 1})
    if any(i < 0 or i >= len(b) for i in locked):
        raise ValueError('锁定帧超出区间。')
    center = (b[:, :2] + b[:, 2:]) / 2
    half = (b[:, 2:] - b[:, :2]) / 2
    q = b.copy() if mode == 'both' else (center.copy() if mode == 'position' else half.copy())
    lower, upper = q - limit, q + limit
    if mode == 'both':
        # Conservatively prevent either edge from crossing the other.
        midpoint = (b[:, :2] + b[:, 2:]) / 2
        upper[:, :2] = np.minimum(upper[:, :2], midpoint - 1e-6)
        lower[:, 2:] = np.maximum(lower[:, 2:], midpoint + 1e-6)
    if mode == 'size':
        lower = np.maximum(lower, 1e-6)
    if image_size is not None:
        dimensions = np.asarray(image_size, dtype=float)
        if mode == 'both':
            lower = np.maximum(lower, 0)
            upper = np.minimum(upper, np.r_[dimensions, dimensions])
        elif mode == 'position':
            lower = np.maximum(lower, half)
            upper = np.minimum(upper, dimensions - half)
        else:
            upper = np.minimum(upper, np.minimum(center, dimensions-center))
        if np.any(lower > upper):
            raise ValueError('框越界，无法满足图像边界约束。')
    lower[locked] = q[locked]
    upper[locked] = q[locked]
    if method == 'interpolate':
        result = np.column_stack([np.interp(np.arange(len(q)), locked, q[locked, j]) for j in range(q.shape[1])])
        result = np.clip(result, lower, upper)
    else:
        # Projected gradient for ||x-q||² + strength*||second_difference(x)||².
        # O(frames) memory/time per iteration; no SciPy or AI dependency.
        result = q.copy()
        step = 1.0 / (1.0 + 16.0 * strength)
        for _ in range(3000):
            second = result[:-2] - 2 * result[1:-1] + result[2:]
            grad = result - q
            grad[:-2] += strength * second
            grad[1:-1] -= 2 * strength * second
            grad[2:] += strength * second
            updated = np.clip(result - step * grad, lower, upper)
            if np.max(np.abs(updated - result)) < 1e-7:
                result = updated
                break
            result = updated
    out = result if mode == 'both' else (
        np.c_[result - half, result + half] if mode == 'position'
        else np.c_[center - result, center + result]
    )
    out[locked] = b[locked]
    if np.max(abs(out - b)) > limit + 1e-6 or np.any(out[:, 2:] <= out[:, :2]):
        raise ValueError('约束无法满足，请缩短区间。')
    return out


def refine_boxes(boxes, anchors, image_size, manual_overrides=None,
                 edge_locks=None, enabled_edges=None, strength=4.0,
                 limit=1.0, mode='both', method='smooth'):
    """Solve independent anchor intervals; explicit manual corrections take priority.

    Automatic movement is bounded around the corrected baseline. Manual changes
    are deliberately not clipped to the automatic limit. Disabled edges retain
    that baseline. Coupled position/size modes expand a one-edge lock to its pair.
    """
    original = np.asarray(boxes, dtype=float)
    working = original.copy()
    count = len(working)
    if count < 3 or working.shape != (count, 4):
        raise ValueError('至少选择三帧有效矩形。')
    if not np.isfinite([strength, limit]).all() or limit < 0 or strength < 0:
        raise ValueError('平滑参数无效。')
    if mode not in ('both', 'position', 'size') or method not in ('smooth', 'interpolate'):
        raise ValueError('未知的平滑模式。')
    manual_overrides = manual_overrides or {}
    edge_locks = edge_locks or {}
    fixed = np.zeros((count, 4), dtype=bool)
    manual = np.zeros_like(fixed)
    for frame, values in manual_overrides.items():
        if not isinstance(frame, int) or not 0 <= frame < count:
            raise ValueError('人工修正帧超出区间。')
        for edge, value in values.items():
            if edge not in range(4) or not np.isfinite(value):
                raise ValueError('人工边界坐标无效。')
            working[frame, edge] = value
            manual[frame, edge] = True
            fixed[frame, edge] = True
    for frame, edges in edge_locks.items():
        if not isinstance(frame, int) or not 0 <= frame < count or any(e not in range(4) for e in edges):
            raise ValueError('边界锁定位置无效。')
        fixed[frame, list(edges)] = True
    locked_frames = sorted(set(anchors) | {0, count-1})
    if any(i < 0 or i >= count for i in locked_frames):
        raise ValueError('锁定帧超出区间。')
    fixed[locked_frames] = True
    dimensions = np.asarray(image_size, dtype=float)
    if (not np.isfinite(working).all() or np.any(working[:, 2:] <= working[:, :2])
            or np.any(working[:, :2] < 0) or np.any(working[:, 2:] > dimensions)):
        raise ValueError('人工修正使矩形越界或宽高无效，请调整边界。')
    enabled = np.asarray([True]*4 if enabled_edges is None else enabled_edges, dtype=bool)
    if enabled.shape != (4,):
        raise ValueError('需要四条边的平滑开关。')
    center = (working[:, :2]+working[:, 2:])/2
    half = (working[:, 2:]-working[:, :2])/2
    if mode == 'both':
        q = working.copy()
        lo, hi = np.maximum(q-limit, 0), np.minimum(q+limit, np.r_[dimensions, dimensions])
        hi[:, :2] = np.minimum(hi[:, :2], center-1e-6)
        lo[:, 2:] = np.maximum(lo[:, 2:], center+1e-6)
        fixed[:, ~enabled] = True
        variable_fixed = fixed
    else:
        if not enabled.all():
            raise ValueError('逐边开关请使用「位置与尺寸」模式。')
        variable_fixed = fixed[:, :2] | fixed[:, 2:]
        fixed[:, :2] = variable_fixed
        fixed[:, 2:] = variable_fixed
        q = center.copy() if mode == 'position' else half.copy()
        lo, hi = q-limit, q+limit
        if mode == 'position':
            lo, hi = np.maximum(lo, half), np.minimum(hi, dimensions-half)
        else:
            lo, hi = np.maximum(lo, 1e-6), np.minimum(hi, np.minimum(center, dimensions-center))
    result = q.copy()
    for edge in range(q.shape[1]):
        knots = np.flatnonzero(variable_fixed[:, edge])
        for start, stop in zip(knots, knots[1:]):
            if stop-start < 2:
                continue
            target = q[start:stop+1, edge]
            lower, upper = lo[start:stop+1, edge].copy(), hi[start:stop+1, edge].copy()
            lower[[0,-1]] = upper[[0,-1]] = target[[0,-1]]
            if method == 'interpolate':
                fitted = np.clip(np.linspace(target[0], target[-1], len(target)), lower, upper)
            else:
                fitted = target.copy()
                step = 1.0/(1+16*strength)
                for _ in range(3000):
                    second = fitted[:-2]-2*fitted[1:-1]+fitted[2:]
                    grad = fitted-target
                    grad[:-2] += strength*second
                    grad[1:-1] -= 2*strength*second
                    grad[2:] += strength*second
                    updated = np.clip(fitted-step*grad, lower, upper)
                    if np.max(abs(updated-fitted)) < 1e-7:
                        fitted = updated
                        break
                    fitted = updated
            result[start:stop+1, edge] = fitted
    out = result if mode == 'both' else (
        np.c_[result-half, result+half] if mode == 'position'
        else np.c_[center-result, center+result]
    )
    out[fixed] = working[fixed]
    if np.max(abs(out-working)) > limit+1e-6 or np.any(out[:, 2:] <= out[:, :2]):
        raise ValueError('约束无法满足，请缩短区间。')
    return out, working, fixed, manual


def build_preview(images, label_paths, identity, anchors, manual_overrides=None,
                  edge_locks=None, enabled_edges=None, **options):
    if len(images) != len(label_paths) or len(images) < 3:
        raise ValueError('请选择至少三帧。')
    paths = [Path(p).resolve() for p in label_paths]
    image_paths = [Path(p).resolve() for p in images]
    if len({p.parent for p in paths}) != 1 or len({p.parent for p in image_paths}) != 1:
        raise ValueError('一次只能处理一个序列文件夹。')
    if len(set(paths)) != len(paths):
        raise ValueError('标注路径重复，已停止。')
    if identity[1] is None:
        raise ValueError('目标没有 ID，请先为它设置唯一的组 ID。')
    rows = []
    for image, path in zip(image_paths, paths):
        if not path.exists():
            raise ValueError(f'{path.name} 缺少标注，请缩短区间或补齐标注。')
        raw = path.read_bytes()
        doc = json.loads(raw.decode('utf-8-sig'))
        matches = [(i, s) for i, s in enumerate(doc.get('shapes', [])) if
                   s.get('shape_type') == 'rectangle' and (s.get('label'), s.get('group_id')) == identity]
        if len(matches) != 1:
            raise ValueError(f'{path.name} 中目标缺失或有多个框。遮挡、跨接缝请分段处理。')
        index, shape = matches[0]
        box = bounds(shape)
        width, height = doc.get('imageWidth'), doc.get('imageHeight')
        if not width or not height or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError(f'{path.name} 的框越界或图片尺寸无效，请先人工修正。')
        rows.append({'image': image, 'path': path, 'raw': raw, 'doc': doc, 'index': index, 'box': box})
    if len({(r['doc']['imageWidth'], r['doc']['imageHeight']) for r in rows}) != 1:
        raise ValueError('帧尺寸不一致，请分段处理。')
    if all(p.stem.isdigit() for p in image_paths):
        numbers = [int(p.stem) for p in image_paths]
        if any(b - a != 1 for a, b in zip(numbers, numbers[1:])):
            raise ValueError('帧编号不连续，请分段处理。')
    old = np.array([r['box'] for r in rows])
    if np.any(abs(np.diff((old[:, 0] + old[:, 2]) / 2)) > rows[0]['doc']['imageWidth'] / 2):
        raise ValueError('目标可能跨越全景接缝，请在接缝前后分别处理。')
    size = (rows[0]['doc']['imageWidth'], rows[0]['doc']['imageHeight'])
    new, baseline, fixed, manual = refine_boxes(
        old, anchors, size, manual_overrides, edge_locks, enabled_edges, **options
    )
    for i, (row, box) in enumerate(zip(rows, new)):
        updated = copy.deepcopy(row['doc'])
        replace_bounds(updated['shapes'][row['index']], box)
        row['new_box'] = box
        row['baseline_box'] = baseline[i]
        row['fixed_edges'] = fixed[i]
        row['manual_edges'] = manual[i]
        row['after'] = (json.dumps(updated, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return rows


def review_flags(rows, limit):
    """Heuristic review prompts, never an assertion of annotation error."""
    flags = {}
    old = np.array([r['box'] for r in rows])
    sizes = old[:, 2:]-old[:, :2]
    for i, row in enumerate(rows):
        reasons = []
        if limit > 0 and np.any(abs(row['new_box']-row['baseline_box']) >= limit-1e-5):
            reasons.append('达到自动改动上限')
        if i and np.any(abs(sizes[i]-sizes[i-1]) > np.maximum(3, .15*sizes[i-1])):
            reasons.append('原框尺寸变化较大')
        if reasons:
            flags[i] = reasons
    return flags


def atomic_write(path, raw):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(raw)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_preview(rows):
    if not rows:
        raise ValueError('请先生成预览。')
    root = rows[0]['path'].parent
    if any(r['path'].parent != root for r in rows):
        raise ValueError('禁止跨文件夹写入。')
    if any(r['path'].read_bytes() != r['raw'] for r in rows):
        raise ValueError('预览后标注被修改，请重新生成预览。')
    changed = [r for r in rows if np.max(abs(r['new_box'] - r['box'])) > 1e-8]
    if not changed:
        raise ValueError('当前设置没有产生坐标变化。')
    backup = root / '.temporal_smoothing_backups' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    receipt = {'status': 'prepared', 'root': str(root), 'files': []}
    for r in changed:
        (backup / r['path'].name).write_bytes(r['raw'])
        receipt['files'].append({'name': r['path'].name, 'before': sha(r['raw']), 'after': sha(r['after'])})
    atomic_write(backup / 'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode('utf-8'))
    written = []
    try:
        for r in changed:
            if r['path'].read_bytes() != r['raw']:
                raise ValueError('标注正在被其他程序修改，操作取消。')
            atomic_write(r['path'], r['after'])
            written.append(r)
        receipt['status'] = 'applied'
        atomic_write(backup / 'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode('utf-8'))
    except Exception:
        for r in reversed(written):
            atomic_write(r['path'], r['raw'])
        raise
    return backup


def latest_backup(root):
    directory = Path(root) / '.temporal_smoothing_backups'
    if directory.exists():
        for p in sorted(directory.glob('*/transaction.json'), reverse=True):
            if json.loads(p.read_text(encoding='utf-8')).get('status') == 'applied':
                return p.parent
    return None


def undo_preview(backup, expected_root):
    backup, root = Path(backup).resolve(), Path(expected_root).resolve()
    if backup.parent != root / '.temporal_smoothing_backups':
        raise ValueError('备份不属于当前标注文件夹。')
    receipt = json.loads((backup / 'transaction.json').read_text(encoding='utf-8'))
    if Path(receipt['root']).resolve() != root or receipt['status'] != 'applied':
        raise ValueError('备份状态无效。')
    writes = []
    for entry in receipt['files']:
        name = entry['name']
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('备份路径无效。')
        p = root / name
        if p.resolve().parent != root:
            raise ValueError('标注路径越出当前目录。')
        after, before = p.read_bytes(), (backup / name).read_bytes()
        if sha(after) != entry['after'] or sha(before) != entry['before']:
            raise ValueError(f'{name} 在应用后被修改，撤销已停止以保护新修改。')
        writes.append((p, before, after))
    done = []
    try:
        for p, before, after in writes:
            if p.read_bytes() != after:
                raise ValueError('标注正在被其他程序修改，撤销已停止。')
            atomic_write(p, before)
            done.append((p, after))
        receipt['status'] = 'undone'
        atomic_write(backup / 'transaction.json', json.dumps(receipt, ensure_ascii=False, indent=2).encode('utf-8'))
    except Exception:
        for p, after in reversed(done):
            atomic_write(p, after)
        raise
    return len(writes)

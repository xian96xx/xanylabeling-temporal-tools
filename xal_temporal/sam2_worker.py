"""Persistent, isolated inference worker. JSON lines in/out; never writes annotations."""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext

# Launch path: `python -I path/to/sam2_worker.py`. Isolated mode ignores PYTHONPATH,
# so the package root is inserted here. Runtime files live under XAL_TEMPORAL_HOME.
PKG_PARENT = Path(__file__).resolve().parents[1]
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))
os.environ.setdefault('PYTHONUTF8', '1')


def home():
    raw = os.environ.get('XAL_TEMPORAL_HOME')
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


def emit(kind, **data):
    print(json.dumps({'type': kind, **data}, ensure_ascii=False), flush=True)


def main():
    import numpy as np
    from PIL import Image
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from xal_temporal.geometry import (merge_directions, mask_box,
        learn_edge_margins, apply_edge_margins, preserve_nearby_outer_edges)
    from xal_temporal.sam2_refine import refine_rotor_interval
    from xal_temporal.cache import TrackingChunkCache
    from xal_temporal.sam2_inspect import inspect_keyframe

    root = home()
    config = json.loads((root/'runtime/tracking_runtime.json').read_text(encoding='utf-8'))
    checkpoint = Path(config['checkpoint'])
    if not checkpoint.is_absolute(): checkpoint = root/checkpoint
    requested = os.environ.get('XANYLABELING_TRACKING_DEVICE', config.get('device', 'auto')).lower()
    device = 'cpu'
    if requested != 'cpu' and torch.cuda.is_available():
        try:
            # A visible GPU can still have an incompatible driver or architecture.
            (torch.ones(1, device='cuda') + 1).sum().item()
            device = 'cuda'
        except RuntimeError as error:
            print(f'CUDA unavailable, using CPU: {error}', file=sys.stderr, flush=True)
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    def precision():
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if device == 'cuda' and torch.cuda.is_bf16_supported() else nullcontext())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    emit('loading', message=('正在加载 SAM 2.1 Small；首次启动需要稍等。' if device == 'cuda'
                             else '正在使用 CPU；计算会较慢，可先选少量帧。'))
    predictor = build_sam2_video_predictor(config['config'], str(checkpoint),
                                          device=device, apply_postprocessing=False)
    emit('ready', device=device_name)
    margin_cache = {}
    chunk_cache = TrackingChunkCache(limit=128)
    cache_enabled = os.environ.get('XANYLABELING_TRACKING_CACHE', '1') != '0'

    for line in sys.stdin:
        try:
            command = json.loads(line)
            jobfile = Path(command['job']).resolve()
            jobroot = jobfile.parent
            if not jobroot.is_relative_to((home()/'runtime/tracking_jobs').resolve()):
                raise ValueError('任务路径不在跟踪工作目录内。')
            job = json.loads(jobfile.read_text(encoding='utf-8'))
            if job.get('operation') == 'inspect_keyframe':
                emit('loading', message='正在检查关键帧的局部识别区域。')
                with torch.inference_mode(), precision():
                    inspection = inspect_keyframe(predictor, job['image'], job['box'], job.get('points', []), jobroot)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                emit('inspection_done', **inspection)
                continue
            images = [Path(p) for p in job['images']]
            anchors = {int(i): b for i, b in job['anchors'].items()}
            rotor = job.get('options', {}).get('boundary_mode') == 'rotor'
            prompts = {int(i): p for i, p in job.get('prompts', {}).items()}
            masks = {int(i): p for i, p in job.get('mask_prompts', {}).items()}
            if any(i not in anchors for i in masks): raise ValueError('涂抹区域必须属于人工关键帧。')
            if any(i not in anchors for i in prompts): raise ValueError('部件提示必须属于人工关键帧。')
            originals = {int(i): b for i, b in job.get('original_boxes', {}).items()}
            total = sum((b-a+1)*int(a in anchors)+(b-a+1)*int(b in anchors)
                        for a, b in job['segments'])
            if rotor or prompts or masks: total *= 2
            completed = 0
            cache_hits = 0
            cached_frames = 0
            started = time.monotonic()
            margins = None
            samples = {}
            calibration_records = []
            if rotor:
                emit('loading', message='正在参考人工框的旋翼／浅灰边范围。')
                with torch.inference_mode(), precision():
                    for index, human in anchors.items():
                        image = images[index]
                        stamp = image.stat()
                        key = (str(image), stamp.st_size, stamp.st_mtime_ns, tuple(human))
                        if key not in margin_cache:
                            with tempfile.TemporaryDirectory(prefix='edge_', dir=jobroot) as folder:
                                dest = Path(folder)/'000000.jpg'
                                if image.suffix.lower() in ('.jpg', '.jpeg'):
                                    try: os.link(image, dest)
                                    except OSError:
                                        import shutil
                                        shutil.copyfile(image, dest)
                                else:
                                    with Image.open(image) as im: im.convert('RGB').save(dest, quality=95)
                                seed_state = predictor.init_state(folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
                                _, _, seed_logits = predictor.add_new_points_or_box(seed_state, 0, 1, box=np.asarray(human, dtype=np.float32))
                                margin_cache[key] = mask_box(seed_logits[0, 0].cpu().float().numpy(), soft=True)
                                del seed_state, seed_logits
                        samples[index] = (human, margin_cache[key])
                gc.collect(); torch.cuda.empty_cache()

            def direction(order, box):
                nonlocal completed, cache_hits, cached_frames
                output, evidence, carry = {}, {}, None
                offset = 0
                while offset < len(order):
                    indices = order[offset:offset+32]
                    paths = [images[i] for i in indices]
                    fingerprints = chunk_cache.fingerprints(paths) if cache_enabled else None
                    cache_key = chunk_cache.key(fingerprints, box, carry, rotor, margins) if cache_enabled else None
                    cached = chunk_cache.get(cache_key) if cache_enabled else None
                    if cached is not None:
                        boxes, cached_evidence, carry = cached
                        for local, index in enumerate(indices):
                            output[index] = boxes[local]
                            if rotor: evidence[index] = cached_evidence[local]
                        completed += len(indices)
                        cache_hits += 1
                        cached_frames += len(indices)
                        emit('progress', value=min(completed,total), total=total,
                             message=f'复用已计算的 {len(indices)} 帧（边界条件未变）')
                        if len(indices) < 32 or offset+len(indices) == len(order): break
                        if not carry.any():
                            for index in order[offset+len(indices):]: output[index] = None
                            break
                        offset += len(indices)-1
                        continue
                    with tempfile.TemporaryDirectory(prefix='chunk_', dir=jobroot) as folder:
                        for local, index in enumerate(indices):
                            source, dest = images[index], Path(folder)/f'{local:06d}.jpg'
                            if source.suffix.lower() in ('.jpg', '.jpeg'):
                                try:
                                    os.link(source, dest)
                                except OSError:
                                    import shutil
                                    shutil.copyfile(source, dest)
                            else:
                                with Image.open(source) as im:
                                    im.convert('RGB').save(dest, quality=95)
                        state = predictor.init_state(folder, offload_video_to_cpu=True,
                                                     offload_state_to_cpu=True)
                        if carry is None:
                            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                                            box=np.asarray(box, dtype=np.float32))
                        else:
                            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=carry)
                        for local, ids, logits in predictor.propagate_in_video(state):
                            mask = (logits[0, 0] > 0).cpu().numpy()
                            extent = (mask_box(logits[0, 0].cpu().float().numpy(), soft=True)
                                      if rotor else mask_box(mask))
                            if rotor: evidence[indices[local]] = extent
                            if rotor:
                                extent = apply_edge_margins(extent, margins, (mask.shape[1], mask.shape[0]))
                            output[indices[local]] = extent
                            if local == len(indices)-1:
                                carry = mask.copy()
                            completed += 1
                            emit('progress', value=min(completed, total), total=total,
                                 message=f'跟踪第 {indices[local]+1} 帧')
                        del state, logits
                        gc.collect()
                        torch.cuda.empty_cache()
                    # Store only completed chunks whose source bytes did not change.
                    if (cache_enabled and carry is not None and
                            all(i in output for i in indices) and
                            chunk_cache.fingerprints(paths) == fingerprints):
                        chunk_cache.put(cache_key, [output[i] for i in indices],
                                        [evidence.get(i) for i in indices], carry)
                    if len(indices) < 32 or offset+len(indices) == len(order):
                        break
                    if carry is None or not carry.any():
                        for index in order[offset+len(indices):]:
                            output[index] = None
                        break
                    offset += len(indices)-1
                return output, evidence

            predictions = {}
            with torch.inference_mode(), precision():
                for a, b in job['segments']:
                    if rotor:
                        # The first human box defines the shared framing
                        # convention; only this interval's endpoints refine it.
                        # A correction in another interval cannot alter these
                        # margins, keeping local recomputation consistent.
                        keys = sorted({min(anchors), a, b}.intersection(anchors))
                        margins = learn_edge_margins([samples[i] for i in keys])
                        calibration_records.append({'start': a, 'end': b, 'margins': margins})
                    left, left_evidence = direction(list(range(a, b+1)), anchors[a]) if a in anchors else ({}, {})
                    right, right_evidence = direction(list(range(b, a-1, -1)), anchors[b]) if b in anchors else ({}, {})
                    combined = merge_directions(left, right, a, b, preserve_edges=rotor)
                    segment_prompts = {i: prompts[i] for i in (a,b) if i in prompts}
                    segment_masks = {i: masks[i] for i in (a,b) if i in masks}
                    if rotor or segment_prompts or segment_masks:
                        if rotor:
                            observed = merge_directions(left_evidence, right_evidence, a, b, preserve_edges=True)
                            for i, item in combined.items(): item['evidence_box'] = observed[i]['box']
                        def local_progress(index):
                            nonlocal completed
                            completed += 1
                            emit('progress', value=min(completed,total), total=total,
                                 message=f'局部精修第 {index+1} 帧的机臂／旋翼外缘')
                        local_margins = refine_rotor_interval(predictor,images,anchors,a,b,
                            combined,jobroot,local_progress,margin_cache,segment_prompts,segment_masks)
                        if rotor: calibration_records[-1]['local_margins'] = local_margins
                    predictions.update(combined)
            for i, box in anchors.items():
                if any(a <= i <= b for a, b in job['segments']):
                    item = predictions.setdefault(i,{})
                    item.update(box=box,reasons=item.get('edge_review_reasons',[]))
            for i, item in predictions.items():
                item['boundary_mode'] = 'rotor' if rotor else 'mask'
                if rotor and i not in anchors:
                    item['model_box'] = item['box']
                    item['box'], protected = preserve_nearby_outer_edges(item['box'], originals.get(i), item.get('evidence_box'))
                    if protected: item.setdefault('notices',[]).append('已保留原框附近的细边范围')
            # A one-frame range needs no propagation.
            if not job['segments']:
                predictions = {i: {'box': b, 'reasons': [], 'boundary_mode': 'rotor' if rotor else 'mask'} for i, b in anchors.items()}
            result = {'predictions': predictions, 'seconds': time.monotonic()-started,
                      'gpu': device_name, 'device': device, 'edge_margins': calibration_records,
                      'cache_hits': cache_hits, 'cached_frames': cached_frames}
            resultfile = jobroot/'result.json'
            resultfile.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            emit('done', result=str(resultfile), seconds=result['seconds'])
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit('error', message=str(error))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit('error', message=str(error))
        sys.exit(1)

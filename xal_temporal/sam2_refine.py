"""Image-supported local video refinement for thin arms and rotor boundaries.

The full-frame tracker supplies identity and crop location. A second video pass
sees a stabilized, enlarged target with context. Source images are only read;
all resized crops live in the isolated worker's temporary job directory.
"""
import gc
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from xal_temporal.geometry import (mask_box, local_crop_window,
    crop_box_coordinates, merge_directions, combine_local_extent,
    learn_edge_margins, apply_edge_margins, interval_reference_aspect, local_size_supported,
    rescue_painted_extent)


def refine_rotor_interval(predictor, images, anchors, start, end, coarse,
                          jobroot, progress, calibration_cache, prompts=None, masks=None):
    prompts = prompts or {}
    masks = masks or {}
    resolution = 512 if prompts or masks else 256
    with Image.open(images[start]) as image:
        size = image.size
    guides = {i: anchors.get(i, item.get('evidence_box', item.get('box')))
              for i, item in coarse.items()}
    windows = {i: local_crop_window(box, size) for i, box in guides.items() if box is not None}

    def save_crop(index, destination, window):
        with Image.open(images[index]) as image:
            image.convert('RGB').crop(tuple(window)).resize(
                (resolution,resolution), Image.Resampling.LANCZOS).save(destination, quality=98)

    # Calibrate against local-resolution evidence, rather than a shrunken
    # full-image mask. The first human example keeps the framing convention.
    keys = sorted({min(anchors), start, end}.intersection(anchors))
    samples = []
    for index in keys:
        human = anchors[index]
        image = Path(images[index])
        stamp = image.stat()
        window = local_crop_window(human, size)
        key = (str(image), stamp.st_size, stamp.st_mtime_ns, tuple(human), 'local-v4', resolution)
        if key not in calibration_cache:
            with tempfile.TemporaryDirectory(prefix='local_edge_', dir=jobroot) as folder:
                save_crop(index, Path(folder)/'000000.jpg', window)
                state = predictor.init_state(folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
                _,_,logits = predictor.add_new_points_or_box(state,0,1,
                    box=np.asarray(crop_box_coordinates(human,window,resolution=resolution),np.float32))
                local = mask_box(logits[0,0].float().cpu().numpy(), soft=True)
                calibration_cache[key] = crop_box_coordinates(local,window,inverse=True,resolution=resolution)
                del state,logits
        samples.append((human,calibration_cache[key]))
    margins = ({'relative':[0.]*4,'pixels':[0.]*4} if prompts or masks else learn_edge_margins(samples))

    def direction(order):
        output, carry, offset = {}, None, 0
        while offset < len(order):
            indices = order[offset:offset+32]
            # A missing full-frame detection never seeds a new local object.
            if any(i not in windows for i in indices):
                first_missing = next(j for j,i in enumerate(indices) if i not in windows)
                indices = indices[:first_missing]
            if not indices: break
            with tempfile.TemporaryDirectory(prefix='local_video_',dir=jobroot) as folder:
                for local,index in enumerate(indices):
                    save_crop(index,Path(folder)/f'{local:06d}.jpg',windows[index])
                state=predictor.init_state(folder,offload_video_to_cpu=True,offload_state_to_cpu=True)
                if carry is None:
                    seed=anchors[indices[0]]
                    window = windows[indices[0]]
                    point_args = {}
                    if indices[0] in prompts:
                        points = np.asarray(prompts[indices[0]], dtype=float)
                        xy = (points[:,:2]-np.asarray(window[:2]))*resolution/(window[2]-window[0])
                        if np.any(xy < 0) or np.any(xy >= resolution):
                            raise ValueError('部件提示点超出局部裁剪范围，请靠近目标重新放点。')
                        point_args = {'points': xy.astype(np.float32), 'labels': points[:,2].astype(np.int32)}
                    if indices[0] in masks:
                        import hashlib
                        record = masks[indices[0]]
                        path = Path(record['path'])
                        raw = path.read_bytes()
                        if hashlib.sha256(raw).hexdigest() != record['sha256']:
                            raise ValueError('涂抹提示已变化，请重新载入预览。')
                        with Image.open(path) as painted:
                            if painted.size != tuple(size) or painted.mode != 'RGBA':
                                raise ValueError('涂抹区域尺寸无效。')
                            alpha = painted.getchannel('A').crop(tuple(window)).resize((resolution,resolution),Image.Resampling.NEAREST)
                            seed_mask = np.asarray(alpha) > 0
                        if not seed_mask.any(): raise ValueError('涂抹目标为空，请补充目标区域。')
                        predictor.add_new_mask(state,0,1,mask=seed_mask)
                    else:
                        predictor.add_new_points_or_box(state,0,1,
                            box=np.asarray(crop_box_coordinates(seed,window,resolution=resolution),np.float32), **point_args)
                else:
                    predictor.add_new_mask(state,0,1,mask=carry)
                for local,_,logits in predictor.propagate_in_video(state):
                    index=indices[local]
                    z=logits[0,0].float().cpu().numpy()
                    hard = crop_box_coordinates(mask_box(z,soft=False),windows[index],inverse=True,resolution=resolution)
                    soft = crop_box_coordinates(mask_box(z,soft=True),windows[index],inverse=True,resolution=resolution)
                    if (prompts or masks) and hard is not None and soft is not None:
                        hard, soft = np.asarray(hard), np.asarray(soft)
                        outward = (soft-hard)*np.array([-1,-1,1,1])
                        output[index] = np.where(outward <= 3., soft, hard).tolist()
                    else:
                        output[index] = soft
                    if local==len(indices)-1: carry=z>0
                    progress(index)
                del state,logits
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            if len(indices)<32 or offset+len(indices)==len(order): break
            if carry is None or not carry.any(): break
            offset+=len(indices)-1
        return output

    left=direction(list(range(start,end+1))) if start in anchors else {}
    right=direction(list(range(end,start-1,-1))) if end in anchors else {}
    merged=merge_directions(left,right,start,end,preserve_edges=True)
    observed = {i: item['box'] for i,item in merged.items()}
    for index,item in coarse.items():
        local=merged[index]['box']
        item['coarse_box']=item['box']
        item['local_evidence_box']=local
        if index in anchors:
            if local is not None:
                outward=(np.asarray(anchors[index])-np.asarray(local))*np.array([1,1,-1,-1])
                if np.any(outward>2.5):
                    item['edge_review_reasons']=['人工框与局部轮廓不一致，请核对外缘']
            continue
        guide=item.get('evidence_box',item['box'])
        if guide is None:
            item['box']=None
            continue
        padded=apply_edge_margins(local,margins,size)
        aspect=interval_reference_aspect(anchors,index,start,end)
        item['reference_aspect']=aspect
        item['coarse_evidence_box']=guide
        supported=local_size_supported(observed,index)
        item['local_neighbour_support']=supported
        item['box'],reasons=combine_local_extent(guide,padded,size,aspect,supported)
        recovered = None
        if masks and any('不一致' in reason for reason in reasons):
            recovered = rescue_painted_extent(guide,local,size,aspect,supported)
            if recovered is not None:
                item['box'] = recovered
                reasons = ['人工涂抹辅助恢复外缘，请检查背景']
                item['painted_recovered'] = True
        if masks and recovered is None and local is not None and not any('不一致' in reason for reason in reasons):
            item['box'] = list(local)
        if prompts or masks: reasons.append('人工提示辅助：请核对细边与背景')
        item['evidence_box'],_=combine_local_extent(guide,local,size,aspect,supported)
        if recovered is not None: item['evidence_box'] = list(recovered)
        item['reasons']=list(dict.fromkeys(item.get('reasons',[])+merged[index]['reasons']+reasons))
        if local is not None:
            edge=(np.asarray(item['box'])-np.asarray(local))*np.array([-1,-1,1,1])
            if np.any(edge>3.): item.setdefault('notices',[]).append('已校验人工框比例并结合两路轮廓')
    return margins

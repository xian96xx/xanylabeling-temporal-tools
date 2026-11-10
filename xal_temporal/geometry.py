"""Dependency-light geometry shared by the GUI and isolated model process."""
import numpy as np


def mask_box(logits, soft=False):
    """Use only soft pixels connected to an actual positive foreground seed.

    The mask used as tracking memory stays at zero threshold. This optional
    display extent admits a narrow low-contrast halo, never a new object when
    the model has no positive foreground at all.
    """
    strong = logits > 0
    yy, xx = np.nonzero(strong)
    if not len(xx): return None
    box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    if not soft: return list(map(float, box))
    h, w = logits.shape
    pad_x = max(2, int(np.ceil((box[2]-box[0])*.08)))
    pad_y = max(2, int(np.ceil((box[3]-box[1])*.08)))
    x0, y0, x1, y1 = max(0, box[0]-pad_x), max(0, box[1]-pad_y), min(w, box[2]+pad_x), min(h, box[3]+pad_y)
    allowed = logits[y0:y1, x0:x1] > -2.0
    region = strong[y0:y1, x0:x1].copy()
    for _ in range(max(pad_x, pad_y)+2):
        p = np.pad(region, 1)
        grown = np.logical_or.reduce([p[dy:dy+region.shape[0], dx:dx+region.shape[1]]
                                      for dy in range(3) for dx in range(3)]) & allowed
        if np.array_equal(grown, region): break
        region = grown
    yy, xx = np.nonzero(region)
    return [float(xx.min()+x0), float(yy.min()+y0), float(xx.max()+x0+1), float(yy.max()+y0+1)]


def learn_edge_margins(samples):
    """Outward margins from confirmed human examples, independently per edge."""
    relative, pixels = [], []
    for human, model in samples:
        if model is None: continue
        model, human = np.asarray(model), np.asarray(human)
        wh = np.maximum(model[2:]-model[:2], 1)
        margins = np.maximum(0, (human-model)*np.array([-1, -1, 1, 1]))
        # A very loose or mismatched prompt must not enlarge every later box.
        if box_iou(human, model) < .65: continue
        rel = np.minimum(margins/np.tile(wh, 2), .12)
        relative.append(rel); pixels.append(np.minimum(margins, np.tile(wh, 2)*.12))
    if not relative: return {'relative': [0.]*4, 'pixels': [0.]*4}
    return {'relative': np.quantile(relative, .8, axis=0).tolist(),
            'pixels': np.quantile(pixels, .8, axis=0).tolist()}


def apply_edge_margins(box, margins, size):
    if box is None: return None
    box = np.asarray(box, dtype=float)
    wh = np.tile(box[2:]-box[:2], 2)
    # Preserve a small blur margin as the object recedes; do not force a fixed
    # universal padding on targets that the human framed tightly.
    pad = np.maximum(np.asarray(margins['relative'])*wh, .5*np.asarray(margins['pixels']))
    result = box+pad*np.array([-1, -1, 1, 1])
    return np.clip(result, 0, np.tile(size, 2)).tolist()


def preserve_nearby_outer_edges(box, original, evidence=None):
    """Avoid shaving a few pixels off a plausible existing box in rotor mode."""
    if box is None or original is None or box_iou(box, original) < .72:
        return box, False
    box, original = np.asarray(box, dtype=float), np.asarray(original, dtype=float)
    tolerance = np.maximum(2., .08*np.tile(np.maximum(box[2:]-box[:2], original[2:]-original[:2]), 2))
    outward = (original-box)*np.array([-1, -1, 1, 1])
    keep = (outward > 0) & (outward <= tolerance)
    if evidence is not None:
        # Keep a plausible old edge when an outward change comes only from
        # estimated padding, rather than newly visible foreground pixels.
        observed_outward = (np.asarray(evidence)-original)*np.array([-1, -1, 1, 1])
        keep |= (outward < 0) & (-outward <= tolerance) & (observed_outward <= .5)
    result = box.copy(); result[keep] = original[keep]
    return result.tolist(), bool(keep.any())


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection / union if union > 0 else 0.0


def merge_directions(left, right, start, end, preserve_edges=False):
    output = {}
    for i in range(start, end+1):
        a, b = left.get(i), right.get(i)
        reasons = []
        if a is not None and b is not None:
            if box_iou(a, b) < .6:
                reasons.append('双向跟踪分歧')
                box = a if i-start <= end-i else b
            else:
                if preserve_edges:
                    # Both masks support the same target. Averaging coordinates
                    # can discard a thin arm seen in only one direction.
                    aa, bb = np.asarray(a), np.asarray(b)
                    box = np.r_[np.minimum(aa[:2], bb[:2]),
                                np.maximum(aa[2:], bb[2:])].tolist()
                    scale = np.tile(np.maximum(aa[2:]-aa[:2], bb[2:]-bb[:2]), 2)
                    if np.any(np.abs(aa-bb) > np.maximum(2., .1*scale)):
                        reasons.append('双向外缘分歧，请核对机臂范围')
                else:
                    t = (i-start)/max(1, end-start)
                    box = ((1-t)*np.asarray(a)+t*np.asarray(b)).tolist()
        else:
            box = a if a is not None else b
            if left and right and box is not None:
                reasons.append('一个方向丢失目标')
        if box is None:
            reasons.append('未检出：不自动补框，已有标注保留')
        output[i] = {'box': box, 'reasons': reasons}
    return output


def local_crop_window(box, size):
    """Square source ROI with context; coordinates are never rounded as boxes."""
    box = np.asarray(box, dtype=float)
    width, height = size
    side = min(width, height, max(128, int(np.ceil(3*np.max(box[2:]-box[:2])))))
    center = (box[:2]+box[2:])/2
    x = int(np.clip(np.floor(center[0]-side/2), 0, width-side))
    y = int(np.clip(np.floor(center[1]-side/2), 0, height-side))
    return [x, y, x+side, y+side]


def crop_box_coordinates(box, window, inverse=False, resolution=256):
    if box is None: return None
    window = np.asarray(window, dtype=float)
    origin = np.tile(window[:2], 2)
    scale = (window[2]-window[0])/resolution
    box = np.asarray(box, dtype=float)
    return (box*scale+origin if inverse else (box-origin)/scale).tolist()


def local_size_supported(sequence, index):
    """Check observed neighbouring masks, never interpolate a missing box."""
    box = sequence.get(index)
    if box is None: return False
    wh = np.asarray(box, float)[2:]-np.asarray(box, float)[:2]
    neighbours = [np.asarray(sequence[j], float)[2:]-np.asarray(sequence[j], float)[:2]
                  for j in range(index-2,index+3)
                  if j != index and sequence.get(j) is not None]
    if len(neighbours) < 2: return False
    reference = np.median(neighbours, axis=0)
    return bool(np.all(wh > 0) and np.all(reference > 0) and
                np.all(wh/reference > .7) and np.all(wh/reference < 1.4))


def combine_local_extent(coarse, local, size, reference_aspect=None,
                         neighbour_support=False):
    """Retain image-supported edges, rejecting a local mask on the background.

    This must never manufacture a detection when the video tracker lost it.
    Rejected/missing local evidence keeps the coarse draft with a review reason.
    """
    if coarse is None:
        return None, ['未检出：不自动补框，已有标注保留']
    if local is None:
        return list(coarse), ['局部精修未检出，请核对外缘']
    coarse, local = np.asarray(coarse, float), np.asarray(local, float)
    a, b = coarse[2:]-coarse[:2], local[2:]-local[:2]
    intersection = np.maximum(0, np.minimum(coarse[2:], local[2:])-np.maximum(coarse[:2], local[:2]))
    contained = np.prod(intersection)/max(1., min(np.prod(a), np.prod(b)))
    shift = np.linalg.norm((coarse[:2]+coarse[2:]-local[:2]-local[2:])/2)
    # A small full-frame mask often sees only the fuselage. Rejecting an
    # enlarged local mask solely for being wider cuts the arms back off.
    # Permit this bounded size disagreement only when adjacent LOCAL masks
    # also observe this size and the coarse foreground is contained in it.
    large = np.any(b > 2.2*np.maximum(a,4.))
    supported_expansion = (neighbour_support and contained >= .85 and
                           np.all(b <= 3.5*np.maximum(a,4.)))
    if (np.any(b <= 0) or contained < .65 or
            shift > max(8., .45*np.linalg.norm(np.maximum(a,b))) or
            (large and not supported_expansion) or np.any(b < .35*a)):
        return coarse.tolist(), ['局部轮廓与跟踪不一致，请核对目标']
    outer = np.r_[np.minimum(coarse[:2], local[:2]), np.maximum(coarse[2:], local[2:])]
    result = outer.copy()
    rejected = []
    if reference_aspect is not None and np.isfinite(reference_aspect) and reference_aspect > 0:
        # Local video sees more pixels of this target. Coarse evidence may
        # restore a missing arm, but must not reintroduce a distant roof/branch.
        # Human aspect is a weak rejection check on LARGE extra extensions,
        # never a forced output size or an interpolated replacement rectangle.
        result = local.copy()
        growth = (local-outer)*np.array([1,1,-1,-1])
        tolerance = np.maximum(2., .12*np.tile(b,2))
        def error(box):
            wh = np.maximum(box[2:]-box[:2], 1e-6)
            return abs(np.log((wh[1]/wh[0])/reference_aspect))
        for edge in np.argsort(growth/tolerance):
            candidate = result.copy(); candidate[edge] = outer[edge]
            if (growth[edge] > tolerance[edge] and error(candidate) > .18 and
                    error(candidate) > error(result)+.08):
                rejected.append(int(edge))
            else:
                result = candidate
    result = np.clip(result, 0, np.tile(size, 2))
    difference = np.abs(local-coarse)
    reasons = []
    if large and supported_expansion:
        reasons.append('局部连续轮廓补回机臂，请核对外缘')
    if rejected:
        reasons.append('已抑制不符人工框比例的额外扩张，请核对外缘')
    if np.any(difference > np.maximum(3., .15*np.tile(np.maximum(a,b),2))):
        reasons.append('局部与全图外缘分歧，请核对机臂范围')
    return result.tolist(), reasons


def rescue_painted_extent(coarse, local, size, reference_aspect, neighbour_support):
    """Conservative preview candidate when painted local evidence was rejected.

    This is not an accuracy check. Require consecutive size support, substantial
    overlap, and a width close to the whole-image target, so roof-sized local
    masks remain rejected. Preserve both sources' measured outer edges.
    """
    if coarse is None or local is None or not neighbour_support or reference_aspect is None:
        return None
    coarse,local=np.asarray(coarse,float),np.asarray(local,float)
    if coarse.shape!=(4,) or local.shape!=(4,) or not np.isfinite(coarse).all() or not np.isfinite(local).all() or not np.isfinite(reference_aspect) or reference_aspect <= 0:
        return None
    cw,ch=coarse[2:]-coarse[:2];lw,lh=local[2:]-local[:2]
    if min(cw,ch,lw,lh)<=0:return None
    intersection=np.maximum(0,np.minimum(coarse[2:],local[2:])-np.maximum(coarse[:2],local[:2]))
    coverage=float(np.prod(intersection)/max(1.,cw*ch))
    height_limit=min(1.8,max(.5,1.8*reference_aspect))
    if not (.7<=lw/cw<=1.3 and coverage>=.7 and lh/cw<=height_limit
            and coarse[1]-local[1]<=.25*cw and local[3]-coarse[3]>max(2.,.05*cw)):
        return None
    outer=np.r_[np.minimum(coarse[:2],local[:2]),np.maximum(coarse[2:],local[2:])]
    return np.clip(outer,0,np.tile(size,2)).tolist()


def interval_reference_aspect(anchors, index, start, end):
    """Use only this interval's human endpoints, preserving local recompute."""
    ratios = {}
    for endpoint in (start,end):
        if endpoint in anchors:
            box = np.asarray(anchors[endpoint],float)
            wh = box[2:]-box[:2]
            if np.all(np.isfinite(wh)) and np.all(wh > 0):
                ratios[endpoint] = wh[1]/wh[0]
    if not ratios: return None
    if len(ratios)==1: return float(next(iter(ratios.values())))
    t = np.clip((index-start)/max(1,end-start),0,1)
    return float(np.exp((1-t)*np.log(ratios[start])+t*np.log(ratios[end])))

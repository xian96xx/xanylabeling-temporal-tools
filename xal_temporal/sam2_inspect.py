"""Inspect one keyframe's SAM local seed mask without changing annotations."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from xal_temporal.geometry import local_crop_window, crop_box_coordinates, mask_box


def inspect_keyframe(predictor, image_path, box, points, jobroot):
    image_path, jobroot = Path(image_path), Path(jobroot)
    with Image.open(image_path) as source:
        size = source.size
        window = local_crop_window(box, size)
        resolution = 512 if points else 256
        with tempfile.TemporaryDirectory(prefix='inspect_', dir=jobroot) as folder:
            dest = Path(folder)/'000000.jpg'
            source.convert('RGB').crop(tuple(window)).resize((resolution,resolution),Image.Resampling.LANCZOS).save(dest,quality=98)
            state = predictor.init_state(folder,offload_video_to_cpu=True,offload_state_to_cpu=True)
            kwargs = {}
            if points:
                arr = np.asarray(points,dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all() or not np.isin(arr[:,2], [0,1]).all():
                    raise ValueError('提示点格式无效。')
                xy=(arr[:,:2]-np.asarray(window[:2]))*resolution/(window[2]-window[0])
                if np.any(xy<0) or np.any(xy>=resolution):
                    raise ValueError('提示点超出局部范围，请调整点位后再查看。')
                kwargs={'points':xy.astype(np.float32),'labels':arr[:,2].astype(np.int32)}
            _,_,logits=predictor.add_new_points_or_box(state,0,1,box=np.asarray(crop_box_coordinates(box,window,resolution=resolution),np.float32),**kwargs)
            mask=(logits[0,0]>0).cpu().numpy()
            del state,logits
    # Save only a transparent visual overlay, never alter input images/annotations.
    visible=Image.fromarray(mask.astype(np.uint8)*255).resize((window[2]-window[0],window[3]-window[1]),Image.Resampling.NEAREST)
    overlay=Image.new('RGBA',size,(0,0,0,0));rgba=Image.new('RGBA',visible.size,(25,235,130,0));rgba.putalpha(visible.point(lambda p: 115 if p else 0));overlay.paste(rgba,tuple(window[:2]))
    path=jobroot/'keyframe_mask.png';overlay.save(path)
    return {'overlay':str(path),'window':window,'resolution':resolution,'mask_pixels':int(mask.sum()),'bbox_in_crop':mask_box(mask),'image_size':size}

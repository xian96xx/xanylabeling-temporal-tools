"""Persistent keyframe masks outside source folders; edited masks are explicit prompts."""
import hashlib,json,os,tempfile
from pathlib import Path
from PIL import Image
from .smoothing import atomic_write
from .tracking import PROJECT_ROOT, prompt_path


def directory(rows,identity): return PROJECT_ROOT/'runtime'/'tracking_masks'/prompt_path(rows,identity).stem

def save_mask(rows,identity,frame,anchor,points,image):
    root=directory(rows,identity);root.mkdir(parents=True,exist_ok=True)
    raw=image if isinstance(image,bytes) else Path(image).read_bytes()
    import io
    with Image.open(io.BytesIO(raw)) as im:
        if im.size != tuple(rows[frame]['size']) or im.mode != 'RGBA' or im.getchannel('A').getbbox() is None:
            raise ValueError('涂抹区域为空或尺寸不匹配，请保留目标区域。')
    digest=hashlib.sha256(raw).hexdigest(); filename=digest+'.png'; path=root/filename
    if not path.exists(): atomic_write(path,raw)
    manifest=root/'masks.json'
    if manifest.exists():data=json.loads(manifest.read_text(encoding='utf8'))
    else:data={'version':1,'identity':list(identity),'entries':{}}
    data['entries'][rows[frame]['image'].name]={'stamp':rows[frame]['image_stamp'],'image_sha256':hashlib.sha256(rows[frame]['image'].read_bytes()).hexdigest(),'box':list(anchor),'points':points,'sha256':digest,'file':filename}
    atomic_write(manifest,json.dumps(data,ensure_ascii=False).encode('utf8'))
    return {'path':str(path),'sha256':digest}


def load_masks(rows,identity,anchors,prompts):
    root=directory(rows,identity);manifest=root/'masks.json'
    if not manifest.exists(): return {}
    data=json.loads(manifest.read_text(encoding='utf8'))
    if data.get('version')!=1 or data.get('identity')!=list(identity):return {}
    names={r['image'].name:i for i,r in enumerate(rows)}; result={}
    for name,entry in data.get('entries',{}).items():
        i=names.get(name); filename=entry.get('file','');digest=entry.get('sha256','')
        if i is None or i not in anchors or entry.get('box')!=list(anchors[i]) or entry.get('points')!=prompts.get(i,[]) or entry.get('stamp')!=rows[i]['image_stamp']:continue
        if hashlib.sha256(rows[i]['image'].read_bytes()).hexdigest()!=entry.get('image_sha256'):continue
        if filename!=digest+'.png' or len(digest)!=64:continue
        path=root/filename
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:result[i]={'path':str(path),'sha256':digest}
    return result

def clear_mask(rows,identity,frame):
    root=directory(rows,identity);manifest=root/'masks.json'
    if not manifest.exists():return
    data=json.loads(manifest.read_text(encoding='utf8'));data.get('entries',{}).pop(rows[frame]['image'].name,None)
    atomic_write(manifest,json.dumps(data,ensure_ascii=False).encode('utf8'))

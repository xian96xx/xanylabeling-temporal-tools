"""Plan optional geometric-disagreement recovery; signals are not defect truth."""
import math

def _risk(v):
    o=v.get('box'); c=v.get('coarse_evidence_box'); l=v.get('local_evidence_box')
    if not all(isinstance(b,(list,tuple)) and len(b)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) for b in (o,c,l)):
        return False
    cw,ch=c[2]-c[0],c[3]-c[1]; lw,lh=l[2]-l[0],l[3]-l[1]; ow,oh=o[2]-o[0],o[3]-o[1]
    if min(cw,ch,lw,lh,ow,oh)<=0:return False
    return (max(c[0]-o[0],o[2]-c[2])>.45*cw or
            (cw>2.3*lw and ow>1.8*lw and lw>40) or
            (ow/oh>7.5 and ow>300))

def plan_recovery(predictions, anchors, start, end):
    """Return suggested one-sided segments; never promote seeds to real anchors."""
    flagged=[i for i in sorted(predictions) if start<=i<=end and _risk(predictions[i])]
    grouped=[]
    for i in flagged:
        if not grouped or i-grouped[-1][-1]>4:grouped.append([i])
        else:grouped[-1].append(i)
    grouped=[g for g in grouped if len(g)>=3]
    merged=[]
    for g in grouped:
        if merged and g[0]-merged[-1][-1]<=10:merged[-1].extend(g)
        else:merged.append(g[:])
    segments=[]; seeds={}
    for g in merged:
        a=max(start,g[0]-8);b=g[-1]+1
        # Missing context, actual human anchors and unbounded spans stay untouched.
        if b>end or b-a>256 or any(a<=i<=b for i in anchors):continue
        if any(i not in predictions for i in range(a,b+1)):continue
        seed=predictions[b].get('box')
        if seed is None or _risk(predictions[b]):continue
        if segments and a<=segments[-1][1]:continue
        segments.append((a,b));seeds[b]=list(seed)
    return segments,seeds

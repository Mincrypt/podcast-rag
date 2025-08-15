from typing import List, Dict, Any

def chunk_by_time(
    segments: List[Dict[str, Any]],
    window_s: float = 45.0,
    overlap_s: float = 7.0,
) -> List[Dict[str, Any]]:
    """
    Build chunks using word timestamps if available; otherwise segment timestamps.
    Returns: [ {"text": "...", "start": float, "end": float}, ...]
    """
    words = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            words.extend(seg["words"])
        else:
            words.append({"word": seg["text"], "start": seg["start"], "end": seg["end"]})

    if not words:
        return []

    tmin = words[0]["start"]
    tmax = words[-1]["end"]

    chunks = []
    cur_start = tmin
    while cur_start < tmax:
        cur_end = cur_start + window_s
        w = [w for w in words if (w["start"] < cur_end and w["end"] > cur_start)]
        text = " ".join([x["word"] for x in w]).strip()
        if text:
            chunks.append({"text": text, "start": max(cur_start, tmin), "end": min(cur_end, tmax)})
        cur_start = cur_end - overlap_s
    return chunks

from typing import List, Dict, Any

def chunk_by_time(
    segments: List[Any],  # faster-whisper segments may be dicts or tuples of (Segment, ...)
    window_s: float = 45.0,
    overlap_s: float = 7.0,
) -> List[Dict[str, Any]]:
    """
    Split a list of faster-whisper segments into overlapping chunks.
    Each seg may be a dict-like with keys or a tuple where seg[0] is the Segment object.
    """
    # Normalize segments to objects with attributes if necessary
    seg_objs = []
    for seg in segments:
        seg_obj = seg[0] if isinstance(seg, tuple) else seg
        seg_objs.append(seg_obj)

    # Build a unified list of word dicts with start/end info
    words = []
    for seg in seg_objs:
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})
        else:
            words.append({"word": seg.text, "start": seg.start, "end": seg.end})

    if not words:
        return []

    # Create overlapping time windows
    chunks = []
    start = words[0]["start"]
    end = start + window_s
    current_text = []
    current_meta = {"start": start, "end": end}

    for w in words:
        if w["start"] < end:
            current_text.append(w["word"])
        else:
            chunks.append({
                "text": " ".join(current_text),
                "start": current_meta["start"],
                "end": current_meta["end"],
            })
            start = end - overlap_s
            end = start + window_s
            current_meta = {"start": start, "end": end}
            current_text = [w["word"]]

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start": current_meta["start"],
            "end": current_meta["end"],
        })

    return chunks

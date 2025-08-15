from typing import List, Dict, Any

def chunk_by_time(
    segments: List[Any],
    window_s: float = 45.0,
    overlap_s: float = 7.0,
) -> List[Dict[str, Any]]:
    """
    Build overlapping chunks from either dict-based segments or whisper Segment objects.
    """
    words = []

    for seg in segments:
        # Dict-based segment (legacy)
        if isinstance(seg, dict):
            if seg.get("words"):
                words.extend(seg["words"])
            else:
                words.append({
                    "word": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                })

        # Whisper Segment object
        elif hasattr(seg, "words") and hasattr(seg, "text") and hasattr(seg, "start") and hasattr(seg, "end"):
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            else:
                words.append({"word": seg.text, "start": seg.start, "end": seg.end})

        # Anything else – skip
        else:
            continue

    if not words:
        return []

    tmin = words[0]["start"]
    tmax = words[-1]["end"]
    chunks = []
    cur_start = tmin

    while cur_start < tmax:
        cur_end = cur_start + window_s
        window_words = [
            w for w in words
            if w["start"] < cur_end and w["end"] > cur_start
        ]
        text = " ".join(w["word"] for w in window_words).strip()
        if text:
            chunks.append({
                "text": text,
                "start": max(cur_start, tmin),
                "end": min(cur_end, tmax),
            })
        cur_start = cur_end - overlap_s

    return chunks

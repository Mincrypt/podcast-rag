from faster_whisper import WhisperModel
from typing import Dict, Any

def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    compute_type: str = "int8",
    vad_filter: bool = True,
    beam_size: int = 5,
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "language": "en",
        "segments": [ { "start": 0.0, "end": 2.3, "text": "...", "words": [ { "word": "hi", "start": 0.1, "end": 0.3}, ... ] }, ... ],
        "duration": float,
        "model_size": model_size
      }
    """
    import time
    t0 = time.time()
    model = WhisperModel(model_size, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        vad_filter=vad_filter,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        temperature=0.0,
    )
    out_segments = []
    for seg in segments:
        item = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        if seg.words:
            item["words"] = [ {"word": w.word.strip(), "start": float(w.start), "end": float(w.end)} for w in seg.words ]
        out_segments.append(item)
    return {
        "language": info.language,
        "segments": out_segments,
        "duration": float(info.duration),
        "model_size": model_size,
        "runtime_sec": time.time() - t0,
    }

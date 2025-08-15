from typing import Optional, List, Dict, Any
import os

def diarize(audio_path: str) -> Optional[list]:
    """
    Returns list of segments like:
      [ {"start": 0.0, "end": 3.1, "speaker": "SPEAKER_0"}, ... ]
    or None if diarization isn't configured.
    """
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        return None
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
        diarization = pipeline(audio_path)
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
        return results
    except Exception as e:
        print("Diarization failed:", e)
        return None

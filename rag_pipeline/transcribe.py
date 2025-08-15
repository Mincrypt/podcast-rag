def transcribe_audio(audio_path, model_size="base", compute_type="int8", vad_filter=True, beam_size=5):
    import time
    from faster_whisper import WhisperModel
    
    t0 = time.time()
    model = WhisperModel(model_size, compute_type=compute_type)
    
    try:
        segments, info = model.transcribe(
            audio_path,
            vad_filter=vad_filter,
            beam_size=beam_size,
        )
        
        # Convert segments to list to check if empty
        segments_list = list(segments)
        
        if not segments_list:
            # Return empty transcription if no segments found
            return {
                'language': 'unknown',
                'duration': 0.0,
                'segments': [],
                'text': '',
                'processing_time': time.time() - t0
            }
        
        # Process segments normally
        transcription_text = " ".join([segment.text for segment in segments_list])
        
        return {
            'language': info.language,
            'duration': info.duration,
            'segments': segments_list,
            'text': transcription_text,
            'processing_time': time.time() - t0
        }
        
    except ValueError as e:
        if "max() arg is an empty sequence" in str(e):
            print(f"Language detection failed for {audio_path}. Retrying with explicit language.")
            
            try:
                # Retry with explicit language and disabled VAD
                segments, info = model.transcribe(
                    audio_path,
                    language="en",  # Force English
                    vad_filter=False,  # Disable VAD filtering
                    beam_size=beam_size,
                )
                
                segments_list = list(segments)
                transcription_text = " ".join([segment.text for segment in segments_list])
                
                return {
                    'language': 'en',
                    'duration': info.duration if hasattr(info, 'duration') else 0.0,
                    'segments': segments_list,
                    'text': transcription_text,
                    'processing_time': time.time() - t0
                }
            except Exception as retry_error:
                print(f"Retry also failed: {retry_error}")
                # Return empty result as fallback
                return {
                    'language': 'unknown',
                    'duration': 0.0,
                    'segments': [],
                    'text': '',
                    'processing_time': time.time() - t0
                }
        else:
            # Re-raise other ValueErrors
            raise e
    except Exception as e:
        print(f"Unexpected error during transcription: {e}")
        # Return empty result as fallback
        return {
            'language': 'unknown',
            'duration': 0.0,
            'segments': [],
            'text': '',
            'processing_time': time.time() - t0
        }

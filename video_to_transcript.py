#!/usr/bin/env python3
"""
Video to Transcript Converter using Whisper
Extracts audio from video and transcribes it using OpenAI Whisper
"""

import os
import sys
import argparse
from pathlib import Path
import whisper
from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to the input video file
        output_audio_path: Optional path for output audio file
        
    Returns:
        Path to the extracted audio file
    """
    if output_audio_path is None:
        video_file = Path(video_path)
        output_audio_path = str(video_file.parent / f"{video_file.stem}_audio.wav")
    
    print(f"Extracting audio from video: {video_path}")
    print(f"Output audio file: {output_audio_path}")
    
    try:
        # Load video and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write audio to file
        audio.write_audiofile(output_audio_path, codec='pcm_s16le', verbose=False, logger=None)
        
        # Clean up
        audio.close()
        video.close()
        
        print(f"✓ Audio extracted successfully: {output_audio_path}")
        return output_audio_path
    
    except Exception as e:
        print(f"✗ Error extracting audio: {str(e)}")
        sys.exit(1)


def transcribe_audio_with_whisper(audio_path: str, model_size: str = "base", language: str = None) -> dict:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (e.g., 'en', 'es', 'fr')
        
    Returns:
        Dictionary containing transcription results
    """
    print(f"\nLoading Whisper model: {model_size}")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"✗ Error loading Whisper model: {str(e)}")
        sys.exit(1)
    
    print(f"Transcribing audio: {audio_path}")
    
    try:
        # Transcribe audio
        if language:
            result = model.transcribe(audio_path, language=language)
        else:
            result = model.transcribe(audio_path)
        
        print("✓ Transcription completed")
        return result
    
    except Exception as e:
        print(f"✗ Error transcribing audio: {str(e)}")
        sys.exit(1)


def save_transcript(result: dict, output_path: str = None, format: str = "txt"):
    """
    Save transcript to file.
    
    Args:
        result: Transcription result from Whisper
        output_path: Path for output file
        format: Output format ('txt', 'srt', 'vtt', 'json')
    """
    if output_path is None:
        output_path = "transcript.txt"
    
    print(f"\nSaving transcript to: {output_path}")
    
    try:
        if format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"✓ Transcript saved as text file")
        
        elif format == "srt":
            # Convert to SRT format
            srt_content = ""
            for i, segment in enumerate(result["segments"], 1):
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"✓ Transcript saved as SRT file")
        
        elif format == "vtt":
            # Convert to VTT format
            vtt_content = "WEBVTT\n\n"
            for segment in result["segments"]:
                start = format_timestamp_vtt(segment["start"])
                end = format_timestamp_vtt(segment["end"])
                text = segment["text"].strip()
                vtt_content += f"{start} --> {end}\n{text}\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(vtt_content)
            print(f"✓ Transcript saved as VTT file")
        
        elif format == "json":
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✓ Transcript saved as JSON file")
        
        else:
            print(f"✗ Unsupported format: {format}")
            return
        
    except Exception as e:
        print(f"✗ Error saving transcript: {str(e)}")
        sys.exit(1)


def format_timestamp(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from video and transcribe using Whisper"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detect if not specified"
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default=None,
        help="Path for extracted audio file (default: <video_name>_audio.wav)"
    )
    parser.add_argument(
        "--transcript-output",
        type=str,
        default=None,
        help="Path for transcript output file (default: transcript.txt)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="txt",
        choices=["txt", "srt", "vtt", "json"],
        help="Output format for transcript (default: txt)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted audio file after transcription"
    )
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"✗ Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Step 1: Extract audio from video
    audio_path = extract_audio_from_video(args.video_path, args.audio_output)
    
    # Step 2: Transcribe audio with Whisper
    result = transcribe_audio_with_whisper(audio_path, args.model, args.language)
    
    # Step 3: Save transcript
    if args.transcript_output is None:
        video_file = Path(args.video_path)
        ext = args.format
        args.transcript_output = str(video_file.parent / f"{video_file.stem}_transcript.{ext}")
    
    save_transcript(result, args.transcript_output, args.format)
    
    # Step 4: Clean up audio file if not keeping it
    if not args.keep_audio:
        try:
            os.remove(audio_path)
            print(f"\n✓ Cleaned up temporary audio file: {audio_path}")
        except:
            pass
    
    # Print summary
    print("\n" + "="*50)
    print("TRANSCRIPTION SUMMARY")
    print("="*50)
    print(f"Video: {args.video_path}")
    print(f"Model: {args.model}")
    print(f"Language: {result.get('language', 'auto-detected')}")
    print(f"Transcript: {args.transcript_output}")
    print(f"Format: {args.format}")
    print(f"\nTranscribed text preview:")
    print("-"*50)
    print(result["text"][:500] + ("..." if len(result["text"]) > 500 else ""))
    print("="*50)


if __name__ == "__main__":
    main()


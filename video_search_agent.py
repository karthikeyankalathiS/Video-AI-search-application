#!/usr/bin/env python3
"""
AI-Powered Video Search Agent
Context-Anchored Video Search with Multimodal Retrieval
Supports text queries and video clip queries to find semantically relevant segments
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import timedelta

# Core dependencies
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sentence_transformers import SentenceTransformer
import torch

# Optional: For visual embeddings and OCR
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except:
    HAS_OCR = False

# For CLIP visual embeddings
try:
    from transformers import CLIPProcessor, CLIPModel
    _HAS_CLIP_AVAILABLE = True
except:
    _HAS_CLIP_AVAILABLE = False
    print("⚠ Warning: CLIP not available. Visual embeddings will be limited.")

# For OpenAI embeddings (better quality)
try:
    from openai import OpenAI
    _HAS_OPENAI_AVAILABLE = True
except:
    _HAS_OPENAI_AVAILABLE = False
    print("⚠ Warning: OpenAI not available. Will use sentence-transformers.")


class VideoSearchAgent:
    """
    AI-powered video search agent that processes videos and enables
    semantic search across a video corpus.
    """
    
    def __init__(self, model_size="base", embedding_model="all-MiniLM-L6-v2", use_openai=False, openai_api_key=None):
        """Initialize the video search agent."""
        print("Initializing Video Search Agent...")
        
        # Load Whisper for transcription
        print(f"Loading Whisper model: {model_size}")
        self.whisper_model = whisper.load_model(model_size)
        
        # Initialize OpenAI client if available and requested
        self.openai_client = None
        self.use_openai_embeddings = False
        if use_openai and _HAS_OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.use_openai_embeddings = True
                    print("✓ Using OpenAI embeddings for better quality")
                except Exception as e:
                    print(f"⚠ Warning: Could not initialize OpenAI: {e}")
                    print("  Falling back to sentence-transformers")
        
        # Load sentence transformer for semantic embeddings (fallback or default)
        if not self.use_openai_embeddings:
            print(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            # Still load for backwards compatibility, but prioritize OpenAI
            self.embedding_model = SentenceTransformer(embedding_model)
        
        # Similarity threshold - filter results below this score
        # Lower threshold = more results (but may include irrelevant ones)
        # Higher threshold = fewer but more relevant results
        # Note: For OpenAI embeddings, use 0.3-0.4. For sentence-transformers, use 0.2-0.25
        if self.use_openai_embeddings:
            self.similarity_threshold = 0.35  # Higher threshold for OpenAI (better quality embeddings)
            self.visual_similarity_threshold = 0.65  # Much higher threshold for CLIP visual embeddings (strict for image search)
        else:
            self.similarity_threshold = 0.2   # Lower threshold for sentence-transformers
            self.visual_similarity_threshold = 0.55  # Higher threshold for CLIP visual embeddings (strict for image search)
        
        # Load CLIP model for visual embeddings (for mute videos and image queries)
        self.clip_model = None
        self.clip_processor = None
        self.has_clip = False
        if _HAS_CLIP_AVAILABLE:
            try:
                print("Loading CLIP model for visual embeddings...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.eval()
                self.has_clip = True
                print("✓ CLIP model loaded")
            except Exception as e:
                print(f"⚠ Warning: Could not load CLIP model: {e}")
                self.has_clip = False
        
        # Video corpus index
        self.video_index = []
        self.embeddings_cache = {}
        
        print("✓ Agent initialized")
    
    def _has_audio_track(self, video_path: str) -> bool:
        """Check if video has an audio track."""
        try:
            video = VideoFileClip(video_path)
            has_audio = video.audio is not None and video.audio.duration > 0
            video.close()
            return has_audio
        except Exception as e:
            print(f"  ⚠ Could not check audio track: {e}")
            return False
    
    def _create_duration_segments(self, duration: float, segment_length: float = 10.0) -> List[Dict]:
        """Create time segments for mute videos based on duration."""
        segments = []
        num_segments = int(duration / segment_length) + 1
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, duration)
            if start < duration:
                segments.append({
                    "id": i,
                    "start": start,
                    "end": end,
                    "text": ""  # Empty text for mute videos
                })
        return segments
    
    def process_video(self, video_path: str, video_id: str = None) -> Dict:
        """
        Process a single video: extract audio (if available), transcribe, create embeddings.
        Supports mute videos with visual-only processing.
        
        Returns:
            Dictionary with video metadata, segments, and embeddings
        """
        if video_id is None:
            video_id = Path(video_path).stem
        
        print(f"\nProcessing video: {video_path}")
        
        # Check if video has audio track
        has_audio = self._has_audio_track(video_path)
        if not has_audio:
            print("  ℹ Video has no audio track (mute video) - using visual-only processing")
        
        # Step 1: Extract audio and transcribe (if audio exists)
        transcript_result = None
        audio_path = None
        if has_audio:
            print("  [1/5] Extracting audio and transcribing...")
            try:
                audio_path = self._extract_audio(video_path)
                transcript_result = self.whisper_model.transcribe(audio_path)
            except Exception as e:
                print(f"  ⚠ Audio processing failed: {e}, falling back to visual-only")
                has_audio = False
                transcript_result = None
        else:
            print("  [1/5] Skipping audio (mute video)")
        
        # Get video duration and create segments
        video = VideoFileClip(video_path)
        video_duration = video.duration
        video.close()
        
        if transcript_result:
            segments = transcript_result["segments"]
            video_language = transcript_result.get("language", "en")
            full_text = transcript_result["text"]
        else:
            # Create segments for mute video
            segments = self._create_duration_segments(video_duration, segment_length=10.0)
            video_language = None
            full_text = ""
        
        # Step 2: Extract key frames for visual analysis
        print("  [2/5] Extracting key frames...")
        key_frames = self._extract_key_frames(video_path, segments)
        
        # Step 3: Detect on-screen text (OCR)
        print("  [3/5] Detecting on-screen text...")
        on_screen_text = self._extract_on_screen_text(key_frames) if HAS_OCR else []
        
        # Step 4: Create visual embeddings (for mute videos and visual search)
        print("  [4/5] Creating visual embeddings...")
        visual_embeddings = self._create_visual_embeddings(key_frames) if self.has_clip and self.clip_model else []
        
        # Step 5: Create multimodal embeddings
        print("  [5/5] Creating multimodal embeddings...")
        segments_with_embeddings = self._create_segment_embeddings(
            segments,
            on_screen_text,
            visual_embeddings,
            has_audio
        )
        
        # Clean up temporary audio
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        video_data = {
            "video_id": video_id,
            "video_path": video_path,
            "duration": video_duration,
            "language": video_language,
            "has_audio": has_audio,
            "segments": segments_with_embeddings,
            "full_text": full_text
        }
        
        self.video_index.append(video_data)
        print(f"  ✓ Video processed: {len(segments_with_embeddings)} segments ({'with audio' if has_audio else 'mute'})")
        
        return video_data
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video."""
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_temp_audio.wav")
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            video.close()
            raise ValueError("Video has no audio track")
        audio.write_audiofile(output_path, codec='pcm_s16le', verbose=False, logger=None)
        audio.close()
        video.close()
        return output_path
    
    def _create_visual_embeddings(self, key_frames: List[Dict]) -> List[np.ndarray]:
        """Generate visual embeddings using CLIP model with normalization."""
        if not self.has_clip or self.clip_model is None:
            return []
        
        visual_embeddings = []
        for frame_data in key_frames:
            try:
                # Convert frame array to PIL Image and ensure RGB format
                frame_image = Image.fromarray(frame_data["frame"]).convert("RGB")
                
                # CLIP processor handles resizing and normalization
                inputs = self.clip_processor(images=frame_image, return_tensors="pt")
                
                # Generate embedding
                with torch.no_grad():
                    visual_features = self.clip_model.get_image_features(**inputs)
                
                embedding = visual_features[0].cpu().numpy()
                
                # Normalize embedding for consistent cosine similarity calculations
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                else:
                    # Invalid embedding, use zero vector (will be skipped in search)
                    embedding = np.zeros(512)
                
                visual_embeddings.append(embedding)
            except Exception as e:
                # Use zero vector if processing fails (will be skipped in search)
                print(f"    ⚠ Visual embedding failed for frame: {e}")
                visual_embeddings.append(np.zeros(512))  # CLIP base model outputs 512-dim
        
        return visual_embeddings
    
    def _extract_key_frames(self, video_path: str, segments: List[Dict], frames_per_segment: int = 1) -> List[Dict]:
        """Extract key frames at segment timestamps."""
        video = VideoFileClip(video_path)
        key_frames = []
        
        for segment in segments:
            # Extract frame at segment midpoint
            mid_time = (segment["start"] + segment["end"]) / 2
            if mid_time < video.duration:
                frame = video.get_frame(mid_time)
                key_frames.append({
                    "timestamp": mid_time,
                    "segment_id": segment.get("id", 0),
                    "frame": frame
                })
        
        video.close()
        return key_frames
    
    def _extract_on_screen_text(self, key_frames: List[Dict]) -> List[Dict]:
        """Extract on-screen text using OCR."""
        if not HAS_OCR:
            return []
        
        on_screen_text = []
        for frame_data in key_frames:
            try:
                frame_image = Image.fromarray(frame_data["frame"])
                text = pytesseract.image_to_string(frame_image)
                if text.strip():
                    on_screen_text.append({
                        "timestamp": frame_data["timestamp"],
                        "segment_id": frame_data["segment_id"],
                        "text": text.strip()
                    })
            except:
                continue
        
        return on_screen_text
    
    def _create_segment_embeddings(self, segments: List[Dict], on_screen_text: List[Dict], 
                                   visual_embeddings: List[np.ndarray] = None, 
                                   has_audio: bool = True) -> List[Dict]:
        """Create multimodal embeddings for each segment."""
        segments_with_embeddings = []
        
        # Create text-to-OCR mapping
        ocr_map = {item["segment_id"]: item["text"] for item in on_screen_text}
        
        # Create segment_id to visual embedding mapping
        visual_emb_map = {}
        if visual_embeddings:
            for i, segment in enumerate(segments):
                if i < len(visual_embeddings):
                    visual_emb_map[segment.get("id", i)] = visual_embeddings[i]
        
        for i, segment in enumerate(segments):
            segment_id = segment.get("id", i)
            
            # Combine transcript text with on-screen text
            transcript_text = segment.get("text", "").strip()
            ocr_text = ocr_map.get(segment_id, "")
            
            # Create combined text for embedding
            combined_text = f"{transcript_text} {ocr_text}".strip()
            
            # Generate text embedding (if text exists)
            text_embedding = None
            if combined_text and has_audio:
                text_embedding = self._get_text_embedding(combined_text)
            elif combined_text and not has_audio and ocr_text:
                # For mute videos, use OCR text if available
                text_embedding = self._get_text_embedding(ocr_text)
            
            # Get visual embedding
            visual_embedding = visual_emb_map.get(segment_id)
            
            # Choose embedding based on availability
            # Note: We don't combine text (1536-dim) and visual (512-dim) embeddings
            # because they're in different embedding spaces with different dimensions.
            # Text embeddings are used for text/audio/video queries.
            # Visual embeddings are used separately for image queries.
            if text_embedding is not None:
                # Use text embedding as primary (for text-based search)
                embedding = text_embedding
            else:
                # For mute videos without text, use zero vector of correct text dimension
                # This ensures dimension compatibility but these segments will have zero similarity
                # in text-based searches (will be filtered out by threshold)
                # Visual search can still use visual_embedding field separately
                if self.use_openai_embeddings:
                    embedding = np.zeros(1536)  # OpenAI text embedding dimension
                else:
                    embedding = np.zeros(384)   # sentence-transformers text embedding dimension
                
                if visual_embedding is None:
                    print(f"    ⚠ Warning: No embedding for segment {segment_id} (no text, no visual)")
            
            segment_data = {
                "id": segment_id,
                "start": segment["start"],
                "end": segment["end"],
                "text": transcript_text,
                "on_screen_text": ocr_text,
                "embedding": embedding.tolist(),
                "combined_text": combined_text
            }
            
            # Store visual embedding separately for visual search
            if visual_embedding is not None:
                segment_data["visual_embedding"] = visual_embedding.tolist()
            
            segments_with_embeddings.append(segment_data)
        
        return segments_with_embeddings
    
    def process_corpus(self, video_paths: List[str]):
        """Process multiple videos to build the corpus index."""
        print(f"\n{'='*70}")
        print(f"Processing Video Corpus ({len(video_paths)} videos)")
        print(f"{'='*70}")
        
        for video_path in video_paths:
            if not os.path.exists(video_path):
                print(f"⚠ Skipping: {video_path} (not found)")
                continue
            self.process_video(video_path)
        
        print(f"\n✓ Corpus processed: {len(self.video_index)} videos indexed")
        print(f"  Total segments: {sum(len(v['segments']) for v in self.video_index)}")
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using OpenAI or sentence-transformers."""
        if self.use_openai_embeddings and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
                return embedding
            except Exception as e:
                print(f"    ⚠ OpenAI embedding failed: {e}, using fallback")
                return self.embedding_model.encode(text)
        else:
            return self.embedding_model.encode(text)
    
    def search_by_text(self, query: str, top_k: int = 5, min_similarity: float = None) -> List[Dict]:
        """
        Search for relevant segments using a text query.
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of ranked segments with similarity scores
        """
        print(f"\nSearching with text query: '{query}'")
        
        # Create query embedding
        query_embedding = self._get_text_embedding(query)
        
        # Use helper method for search
        min_sim = min_similarity if min_similarity is not None else self.similarity_threshold
        return self._search_with_embedding(query_embedding, top_k, min_similarity=min_sim)
    
    def search_by_video_clip(self, clip_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant segments using a sample video clip.
        
        Args:
            clip_path: Path to the query video clip
            top_k: Number of top results to return
            
        Returns:
            List of ranked segments with similarity scores
        """
        print(f"\nSearching with video clip: {clip_path}")
        
        # Process the query clip
        clip_data = self.process_video(clip_path, video_id="query_clip")
        
        if not clip_data["segments"]:
            return []
        
        # Use the first segment's embedding as query (or average of all segments)
        clip_embeddings = [np.array(s["embedding"]) for s in clip_data["segments"]]
        query_embedding = np.mean(clip_embeddings, axis=0)
        
        # Normalize query embedding for efficient cosine similarity calculation
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Remove query clip from index first
        self.video_index = [v for v in self.video_index if v["video_id"] != "query_clip"]
        
        # Search across corpus with similarity threshold filtering
        min_similarity = self.similarity_threshold
        results = []
        for video_data in self.video_index:
            for segment in video_data["segments"]:
                segment_embedding = np.array(segment["embedding"])
                
                # Normalize segment embedding
                segment_embedding = segment_embedding / (np.linalg.norm(segment_embedding) + 1e-8)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, segment_embedding)
                
                # Filter by similarity threshold to remove irrelevant results
                if similarity < min_similarity:
                    continue
                
                results.append({
                    "video_id": video_data["video_id"],
                    "video_path": video_data["video_path"],
                    "segment": segment,
                    "similarity": float(similarity),
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        if not results:
            print(f"  ⚠ No results found above similarity threshold {min_similarity:.2f}")
            return []
        
        print(f"  ✓ Found {len(results)} relevant segment(s) (threshold: {min_similarity:.2f})")
        return results[:top_k]
    
    def search_by_audio(self, audio_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant segments using an audio query.
        
        Args:
            audio_path: Path to the query audio file
            top_k: Number of top results to return
            
        Returns:
            List of ranked segments with similarity scores
        """
        print(f"\nSearching with audio query: {audio_path}")
        
        # Transcribe audio to text
        print("  Transcribing audio...")
        try:
            transcript_result = self.whisper_model.transcribe(audio_path)
            query_text = transcript_result["text"]
            print(f"  Transcription: {query_text[:100]}...")
        except Exception as e:
            print(f"  ✗ Error transcribing audio: {e}")
            return []
        
        if not query_text.strip():
            print("  ✗ No speech detected in audio")
            return []
        
        # Generate text embedding from transcription
        query_embedding = self._get_text_embedding(query_text)
        
        # Search using text embedding (same as text search)
        return self._search_with_embedding(query_embedding, top_k, min_similarity=self.similarity_threshold)
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant segments using an image query.
        
        Args:
            image_path: Path to the query image file
            top_k: Number of top results to return
            
        Returns:
            List of ranked segments with similarity scores
        """
        print(f"\nSearching with image query: {image_path}")
        
        if not self.has_clip or self.clip_model is None:
            print("  ✗ CLIP model not available. Cannot search by image.")
            return []
        
        # Load and process image with consistent preprocessing
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Ensure image is properly sized and normalized (CLIP processor handles this)
            # CLIP processor resizes to 224x224 and normalizes, so query and indexed images match
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Generate visual embedding
            with torch.no_grad():
                query_embedding = self.clip_model.get_image_features(**inputs)
            query_embedding = query_embedding[0].cpu().numpy()
            
            # Normalize embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm < 1e-8:
                print("  ✗ Invalid query embedding (zero vector)")
                return []
            query_embedding = query_embedding / query_norm
        except Exception as e:
            print(f"  ✗ Error processing image: {e}")
            return []
        
        # Search using visual similarity
        # Note: Only use visual embeddings (512-dim), not text embeddings (1536-dim)
        # They're in different embedding spaces and can't be compared
        min_similarity = self.visual_similarity_threshold
        results = []
        
        for video_data in self.video_index:
            for segment in video_data["segments"]:
                # Only use visual embedding for image search (same embedding space)
                if "visual_embedding" not in segment:
                    # Skip segments without visual embeddings (can't compare with image query)
                    continue
                
                segment_embedding = np.array(segment["visual_embedding"])
                
                # Normalize segment embedding (ensure both query and segment are normalized)
                segment_norm = np.linalg.norm(segment_embedding)
                if segment_norm < 1e-8:
                    # Skip invalid embeddings (zero vectors)
                    continue
                segment_embedding = segment_embedding / segment_norm
                
                # Calculate cosine similarity (both are normalized 512-dim CLIP embeddings)
                similarity = np.dot(query_embedding, segment_embedding)
                
                # Apply stricter filtering: only include results above threshold
                # CLIP cosine similarity ranges from -1 to 1, but typically positive for similar images
                if similarity < min_similarity:
                    continue
                
                results.append({
                    "video_id": video_data["video_id"],
                    "video_path": video_data["video_path"],
                    "segment": segment,
                    "similarity": float(similarity),
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        if not results:
            print(f"  ⚠ No results found above visual similarity threshold {min_similarity:.2f}")
            print(f"     Try: Lowering threshold or using a more similar image")
            return []
        
        # Strict quality filter: CLIP similarities for truly similar images are typically 0.70-0.90+
        # Scores below 0.70 are often false positives for unrelated images
        MIN_QUALITY_SIMILARITY = 0.70  # Absolute minimum for a truly relevant match
        
        if not results:
            print(f"  ⚠ No results found above visual similarity threshold {min_similarity:.2f}")
            return []
        
        best_similarity = results[0]["similarity"]
        
        # Quality check 1: Top result must meet absolute quality threshold
        if best_similarity < MIN_QUALITY_SIMILARITY:
            print(f"  ⚠ Best match similarity ({best_similarity:.3f}) is below quality threshold ({MIN_QUALITY_SIMILARITY:.2f})")
            print(f"     For truly similar images, CLIP typically scores 0.70-0.90+")
            print(f"     This image likely doesn't match any video content.")
            return []
        
        # Quality check 2: Top result must be significantly above the threshold
        quality_gap = 0.08  # Require at least 0.08 above threshold
        if best_similarity < min_similarity + quality_gap:
            print(f"  ⚠ Best match similarity ({best_similarity:.3f}) is too close to threshold ({min_similarity:.2f})")
            print(f"     Required minimum: {min_similarity + quality_gap:.2f}")
            print(f"     This image likely doesn't match any video content.")
            return []
        
        print(f"  ✓ Found {len(results)} visually similar segment(s) (threshold: {min_similarity:.2f})")
        print(f"     Best match similarity: {best_similarity:.3f} (quality threshold: {MIN_QUALITY_SIMILARITY:.2f})")
        
        # Return top_k results
        return results[:top_k]
    
    def _search_with_embedding(self, query_embedding: np.ndarray, top_k: int = 5, min_similarity: float = None) -> List[Dict]:
        """Helper method to search with a query embedding."""
        if min_similarity is None:
            min_similarity = self.similarity_threshold
        
        results = []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_dim = len(query_embedding)
        
        # Expected text embedding dimensions
        expected_text_dim = 1536 if self.use_openai_embeddings else 384
        
        for video_data in self.video_index:
            for segment in video_data["segments"]:
                # Skip if no embedding field
                if "embedding" not in segment:
                    continue
                    
                segment_embedding = np.array(segment["embedding"])
                
                # Skip segments with wrong dimension (e.g., visual embeddings in text search)
                if len(segment_embedding) != query_dim:
                    # This segment has wrong dimension (likely a visual embedding)
                    # Skip it - it should only be used in visual/image search
                    continue
                
                # Skip zero vectors (mute videos without text)
                if np.linalg.norm(segment_embedding) < 1e-8:
                    continue
                
                segment_norm = segment_embedding / (np.linalg.norm(segment_embedding) + 1e-8)
                
                # Calculate cosine similarity
                similarity = np.dot(query_norm, segment_norm)
                
                # Filter by minimum similarity threshold
                if similarity < min_similarity:
                    continue
                
                results.append({
                    "video_id": video_data["video_id"],
                    "video_path": video_data["video_path"],
                    "segment": segment,
                    "similarity": float(similarity),
                    "start": segment["start"],
                    "end": segment["end"]
                })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def generate_highlight_reel(self, search_results: List[Dict], output_path: str, 
                                 max_duration: int = 60) -> str:
        """
        Generate a highlight reel from search results.
        
        Args:
            search_results: List of ranked segments
            output_path: Path for output video
            max_duration: Maximum duration in seconds
            
        Returns:
            Path to generated highlight reel
        """
        print(f"\nGenerating highlight reel from {len(search_results)} segments...")
        
        clips = []
        total_duration = 0
        video_cache = {}  # Cache videos to avoid reloading
        
        for result in search_results:
            if total_duration >= max_duration:
                break
            
            video_path = result["video_path"]
            start_time = result["start"]
            end_time = result["end"]
            
            # Limit segment duration
            segment_duration = min(end_time - start_time, 10)  # Max 10 seconds per segment
            end_time = start_time + segment_duration
            
            try:
                # Use cached video or load new one
                if video_path not in video_cache:
                    video_cache[video_path] = VideoFileClip(video_path)
                
                video = video_cache[video_path]
                clip = video.subclip(start_time, end_time)
                clips.append(clip)
                total_duration += segment_duration
            except Exception as e:
                print(f"  ⚠ Error extracting segment: {e}")
                continue
        
        if not clips:
            print("  ✗ No valid clips to combine")
            return None
        
        # Concatenate clips
        print(f"  Combining {len(clips)} clips ({total_duration:.1f}s total)...")
        
        try:
            final_clip = concatenate_videoclips(clips, method="compose")
        except Exception as e:
            print(f"  ⚠ Error concatenating clips: {e}")
            # Try without audio
            clips_no_audio = [c.without_audio() if c.audio else c for c in clips]
            final_clip = concatenate_videoclips(clips_no_audio, method="compose")
        
        # Write output
        try:
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
        except Exception as e:
            # Fallback: write without audio if audio encoding fails
            print(f"  ⚠ Audio encoding failed, writing video without audio: {e}")
            try:
                final_clip_no_audio = final_clip.without_audio()
                final_clip_no_audio.write_videofile(
                    output_path,
                    codec='libx264',
                    audio=False,
                    verbose=False,
                    logger=None
                )
                final_clip_no_audio.close()
            except Exception as e2:
                print(f"  ✗ Failed to write video: {e2}")
                return None
        
        # Clean up
        final_clip.close()
        for clip in clips:
            clip.close()
        for video in video_cache.values():
            video.close()
        
        print(f"  ✓ Highlight reel saved: {output_path}")
        return output_path
    
    def save_index(self, output_path: str):
        """Save the video index to disk."""
        # Convert numpy arrays to lists for JSON serialization
        index_data = []
        for video_data in self.video_index:
            video_copy = video_data.copy()
            index_data.append(video_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Index saved: {output_path}")
    
    def load_index(self, index_path: str):
        """Load video index from disk."""
        with open(index_path, 'r', encoding='utf-8') as f:
            self.video_index = json.load(f)
        
        print(f"✓ Index loaded: {len(self.video_index)} videos")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Video Search Agent - Context-Anchored Video Search"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Process videos and build index')
    index_parser.add_argument('videos', nargs='+', help='Video files to process')
    index_parser.add_argument('--output', '-o', default='video_index.json', help='Output index file')
    index_parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'])
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search videos with text query')
    search_parser.add_argument('query', help='Text query')
    search_parser.add_argument('--index', '-i', default='video_index.json', help='Index file')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--highlight', help='Generate highlight reel (output path)')
    search_parser.add_argument('--max-duration', type=int, default=60, help='Max highlight duration (seconds)')
    
    # Search by clip command
    clip_parser = subparsers.add_parser('search-clip', help='Search videos with video clip')
    clip_parser.add_argument('clip', help='Query video clip path')
    clip_parser.add_argument('--index', '-i', default='video_index.json', help='Index file')
    clip_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    clip_parser.add_argument('--highlight', help='Generate highlight reel (output path)')
    clip_parser.add_argument('--max-duration', type=int, default=60, help='Max highlight duration (seconds)')
    
    # Search by audio command
    audio_parser = subparsers.add_parser('search-audio', help='Search videos with audio query')
    audio_parser.add_argument('audio', help='Query audio file path')
    audio_parser.add_argument('--index', '-i', default='video_index.json', help='Index file')
    audio_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    audio_parser.add_argument('--highlight', help='Generate highlight reel (output path)')
    audio_parser.add_argument('--max-duration', type=int, default=60, help='Max highlight duration (seconds)')
    
    # Search by image command
    image_parser = subparsers.add_parser('search-image', help='Search videos with image query')
    image_parser.add_argument('image', help='Query image file path')
    image_parser.add_argument('--index', '-i', default='video_index.json', help='Index file')
    image_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    image_parser.add_argument('--highlight', help='Generate highlight reel (output path)')
    image_parser.add_argument('--max-duration', type=int, default=60, help='Max highlight duration (seconds)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize agent
    agent = VideoSearchAgent(model_size=getattr(args, 'model', 'base'))
    
    if args.command == 'index':
        # Process videos and build index
        agent.process_corpus(args.videos)
        agent.save_index(args.output)
        
    elif args.command == 'search':
        # Load index and search
        if os.path.exists(args.index):
            agent.load_index(args.index)
        else:
            print(f"✗ Index file not found: {args.index}")
            print("  Run 'index' command first to build the index")
            sys.exit(1)
        
        # Search
        results = agent.search_by_text(args.query, top_k=args.top_k)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*70}")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
            print(f"    Video: {result['video_id']}")
            print(f"    Time: {result['start']:.1f}s - {result['end']:.1f}s")
            print(f"    Text: {result['segment']['text'][:100]}...")
        
        # Generate highlight reel if requested
        if args.highlight and results:
            agent.generate_highlight_reel(results, args.highlight, args.max_duration)
    
    elif args.command == 'search-clip':
        # Load index and search by clip
        if os.path.exists(args.index):
            agent.load_index(args.index)
        else:
            print(f"✗ Index file not found: {args.index}")
            print("  Run 'index' command first to build the index")
            sys.exit(1)
        
        # Search
        results = agent.search_by_video_clip(args.clip, top_k=args.top_k)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*70}")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
            print(f"    Video: {result['video_id']}")
            print(f"    Time: {result['start']:.1f}s - {result['end']:.1f}s")
            print(f"    Text: {result['segment']['text'][:100]}...")
        
        # Generate highlight reel if requested
        if args.highlight and results:
            agent.generate_highlight_reel(results, args.highlight, args.max_duration)
    
    elif args.command == 'search-audio':
        # Load index and search by audio
        if os.path.exists(args.index):
            agent.load_index(args.index)
        else:
            print(f"✗ Index file not found: {args.index}")
            print("  Run 'index' command first to build the index")
            sys.exit(1)
        
        # Check if audio file exists
        if not os.path.exists(args.audio):
            print(f"✗ Audio file not found: {args.audio}")
            sys.exit(1)
        
        # Search
        results = agent.search_by_audio(args.audio, top_k=args.top_k)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*70}")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
            print(f"    Video: {result['video_id']}")
            print(f"    Time: {result['start']:.1f}s - {result['end']:.1f}s")
            text_preview = result['segment'].get('text', '')[:100] or '(mute video)'
            print(f"    Text: {text_preview}...")
        
        # Generate highlight reel if requested
        if args.highlight and results:
            agent.generate_highlight_reel(results, args.highlight, args.max_duration)
    
    elif args.command == 'search-image':
        # Load index and search by image
        if os.path.exists(args.index):
            agent.load_index(args.index)
        else:
            print(f"✗ Index file not found: {args.index}")
            print("  Run 'index' command first to build the index")
            sys.exit(1)
        
        # Check if image file exists
        if not os.path.exists(args.image):
            print(f"✗ Image file not found: {args.image}")
            sys.exit(1)
        
        # Search
        results = agent.search_by_image(args.image, top_k=args.top_k)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*70}")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
            print(f"    Video: {result['video_id']}")
            print(f"    Time: {result['start']:.1f}s - {result['end']:.1f}s")
            text_preview = result['segment'].get('text', '')[:100] or '(visual similarity)'
            print(f"    Text: {text_preview}...")
        
        # Generate highlight reel if requested
        if args.highlight and results:
            agent.generate_highlight_reel(results, args.highlight, args.max_duration)


if __name__ == "__main__":
    main()


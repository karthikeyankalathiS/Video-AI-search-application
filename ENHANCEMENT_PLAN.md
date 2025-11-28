# Enhancement Plan: Multi-Modal Query Support & Mute Video Handling

## Problem Statement Summary

1. **50 indexed videos** in corpus
2. **Query inputs can be**: Text, Audio, Image, or Video
3. **Mute videos** (no audio) must be indexable and searchable
4. **Output**: Relevant segments from the indexed video corpus

---

## Current Limitations

### ❌ Missing Features
1. **Audio Query Support**: Cannot search using audio file input
2. **Image Query Support**: Cannot search using image file input  
3. **Mute Video Handling**: Fails when video has no audio track
4. **Visual-Only Processing**: No fallback when audio is missing

### ⚠️ Current Issues
- `process_video()` always extracts audio → fails on mute videos
- Only text and video clip queries supported
- No visual embeddings for mute video segments
- Missing query routing for different input types

---

## Solution Architecture

### Enhanced Query Processing Pipeline

```
Input (Text/Audio/Image/Video)
    ↓
Query Router → Detect Input Type
    ↓
┌───────────────────────────────────────┐
│  TEXT QUERY                           │
│  → Sentence Transformer Embedding     │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  AUDIO QUERY                          │
│  → Whisper Transcription              │
│  → Sentence Transformer Embedding     │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  IMAGE QUERY                          │
│  → CLIP Image Encoder                 │
│  → Visual Embedding                   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  VIDEO QUERY                          │
│  → Extract Frames + Audio (if exists) │
│  → CLIP + Text Embeddings             │
│  → Combined Embedding                 │
└───────────────────────────────────────┘
    ↓
Unified Query Embedding
    ↓
Semantic Search (Cosine Similarity)
    ↓
Ranked Results
```

### Enhanced Indexing Pipeline

```
Video Input
    ↓
Check for Audio Track
    ├─ YES → Extract Audio → Whisper Transcription
    └─ NO  → Skip Audio Processing (Mute Video)
    ↓
Extract Visual Frames (Always)
    ↓
Extract OCR Text (If Available)
    ↓
Create Multimodal Embeddings
    ├─ Text Embedding (if audio/OCR exists)
    ├─ Visual Embedding (always - CLIP model)
    └─ Combined Embedding
    ↓
Store in Index
```

---

## Implementation Plan

### Phase 1: Mute Video Support ✅

#### 1.1 Detect Audio Availability
- Check if video has audio track before extraction
- Gracefully handle missing audio
- Use visual-only processing for mute videos

#### 1.2 Visual Embedding Model
- Add CLIP (Contrastive Language-Image Pre-training) model
- Generate visual embeddings from frames
- Works independently of audio

#### 1.3 Enhanced Segment Processing
- Store separate embeddings: text, visual, combined
- Support visual-only segments for mute videos

**Files to Modify:**
- `video_search_agent.py` - `process_video()`, `_extract_audio()`, `_create_segment_embeddings()`

**New Dependencies:**
```python
transformers>=4.30.0  # For CLIP model
torchvision>=0.15.0   # For image preprocessing
```

---

### Phase 2: Audio Query Support ✅

#### 2.1 Audio Query Processing
- Accept audio file (mp3, wav, m4a, etc.)
- Transcribe using Whisper
- Generate text embedding from transcription
- Search corpus

#### 2.2 API Endpoint
- `/api/search-audio` - Accept audio file upload
- Process audio → text → embedding → search

**Files to Modify:**
- `video_search_agent.py` - Add `search_by_audio()` method
- `app.py` - Add `/api/search-audio` route
- `templates/index.html` - Add Audio Search tab
- `static/js/app.js` - Add audio search handler

---

### Phase 3: Image Query Support ✅

#### 3.1 Image Query Processing
- Accept image file (jpg, png, etc.)
- Use CLIP to generate visual embedding
- Compare with video frame embeddings
- Return visually similar segments

#### 3.2 Visual Similarity Search
- Extract embeddings from query image
- Compare with all frame embeddings in corpus
- Rank by visual similarity

**Files to Modify:**
- `video_search_agent.py` - Add `search_by_image()`, visual embedding extraction
- `app.py` - Add `/api/search-image` route
- `templates/index.html` - Add Image Search tab
- `static/js/app.js` - Add image search handler

---

### Phase 4: Unified Query Router ✅

#### 4.1 Smart Query Detection
- Auto-detect input type from file extension/MIME type
- Route to appropriate processing pipeline
- Unified search interface

#### 4.2 Multi-Modal Search
- Combine multiple query types (if needed)
- Weighted similarity scores
- Unified ranking

**Files to Modify:**
- `video_search_agent.py` - Add `search()` unified method
- `app.py` - Add `/api/search-unified` route

---

### Phase 5: Scalability for 50 Videos ✅

#### 5.1 Batch Processing
- Process videos in parallel during indexing
- Use multiprocessing for CPU-intensive tasks
- Progress tracking for large batches

#### 5.2 Efficient Storage
- Consider vector database (FAISS) for embeddings
- Compress embeddings
- Lazy loading of video files

#### 5.3 Index Optimization
- Separate indexes for different modalities
- Fast approximate nearest neighbor search
- Caching frequently accessed embeddings

**Files to Modify:**
- `video_search_agent.py` - Add parallel processing, vector DB support
- Consider new file: `vector_index.py` for FAISS integration

---

## Technical Details

### 1. Audio Detection & Mute Video Handling

```python
def has_audio_track(video_path: str) -> bool:
    """Check if video has audio track."""
    video = VideoFileClip(video_path)
    has_audio = video.audio is not None
    video.close()
    return has_audio

def process_video(self, video_path: str, video_id: str = None) -> Dict:
    """Enhanced processing with mute video support."""
    # Check for audio
    has_audio = self._has_audio_track(video_path)
    
    # Process audio if available
    transcript_result = None
    if has_audio:
        audio_path = self._extract_audio(video_path)
        transcript_result = self.whisper_model.transcribe(audio_path)
        os.remove(audio_path)
    
    # Always extract frames (for mute videos too)
    if transcript_result:
        segments = transcript_result["segments"]
    else:
        # Create segments from video duration for mute videos
        video = VideoFileClip(video_path)
        duration = video.duration
        segments = self._create_duration_segments(duration, segment_length=10)
        video.close()
    
    # Extract frames
    key_frames = self._extract_key_frames(video_path, segments)
    
    # Generate visual embeddings (always)
    visual_embeddings = self._create_visual_embeddings(key_frames)
    
    # Create multimodal embeddings
    segments_with_embeddings = self._create_segment_embeddings(
        segments, visual_embeddings, transcript_result
    )
    
    return video_data
```

### 2. Visual Embedding with CLIP

```python
from transformers import CLIPProcessor, CLIPModel
import torch

class VideoSearchAgent:
    def __init__(self, ...):
        # Existing models
        self.whisper_model = whisper.load_model(model_size)
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Add CLIP for visual embeddings
        print("Loading CLIP model for visual embeddings...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
    def _create_visual_embeddings(self, key_frames: List[Dict]) -> List[np.ndarray]:
        """Generate visual embeddings using CLIP."""
        embeddings = []
        for frame_data in key_frames:
            frame_image = Image.fromarray(frame_data["frame"])
            inputs = self.clip_processor(images=frame_image, return_tensors="pt")
            with torch.no_grad():
                visual_features = self.clip_model.get_image_features(**inputs)
            embeddings.append(visual_features[0].numpy())
        return embeddings
```

### 3. Audio Query Processing

```python
def search_by_audio(self, audio_path: str, top_k: int = 5) -> List[Dict]:
    """Search using audio query."""
    # Transcribe audio
    transcript_result = self.whisper_model.transcribe(audio_path)
    query_text = transcript_result["text"]
    
    # Generate text embedding
    query_embedding = self.embedding_model.encode(query_text)
    
    # Search corpus
    return self._search_with_embedding(query_embedding, top_k)
```

### 4. Image Query Processing

```python
def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict]:
    """Search using image query."""
    # Load and process image
    image = Image.open(image_path)
    inputs = self.clip_processor(images=image, return_tensors="pt")
    
    # Generate visual embedding
    with torch.no_grad():
        query_embedding = self.clip_model.get_image_features(**inputs)
    query_embedding = query_embedding[0].numpy()
    
    # Search using visual similarity
    results = []
    for video_data in self.video_index:
        for segment in video_data["segments"]:
            # Use visual embedding if available
            if "visual_embedding" in segment:
                visual_emb = np.array(segment["visual_embedding"])
                similarity = cosine_similarity(query_embedding, visual_emb)
                results.append({
                    "video_id": video_data["video_id"],
                    "similarity": float(similarity),
                    "segment": segment,
                    "start": segment["start"],
                    "end": segment["end"]
                })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
```

---

## Updated Requirements

### New Dependencies

```txt
# Add to requirements.txt
transformers>=4.30.0          # CLIP model
torchvision>=0.15.0           # Image preprocessing
Pillow>=10.0.0                # Already have, but ensure version
```

### Optional (for scalability)
```txt
faiss-cpu>=1.7.4              # Vector database for fast search
# OR
faiss-gpu>=1.7.4              # GPU-accelerated vector search
```

---

## File Structure Changes

### New Files
```
videoAi/
├── visual_embedder.py        # CLIP-based visual embedding utilities
├── query_router.py           # Smart query type detection and routing
└── vector_index.py           # FAISS-based vector index (optional, for scale)
```

### Modified Files
```
videoAi/
├── video_search_agent.py     # Major enhancements
├── app.py                    # New API endpoints
├── templates/index.html      # New UI tabs
├── static/js/app.js          # New query handlers
└── requirements.txt          # New dependencies
```

---

## Implementation Priority

### Priority 1: Critical (Mute Video Support)
1. ✅ Detect audio availability
2. ✅ Visual embedding with CLIP
3. ✅ Mute video processing pipeline
4. ✅ Store visual embeddings separately

### Priority 2: High (Audio & Image Queries)
5. ✅ Audio query processing
6. ✅ Image query processing
7. ✅ API endpoints for new query types
8. ✅ UI updates for new query types

### Priority 3: Medium (Optimization)
9. ⚠️ Parallel video processing
10. ⚠️ Vector database integration
11. ⚠️ Query routing optimization

---

## Testing Strategy

### Unit Tests
- Audio detection function
- Visual embedding generation
- Query type detection
- Mute video processing

### Integration Tests
- End-to-end mute video indexing
- Audio query → search → results
- Image query → search → results
- Mixed corpus (videos with/without audio)

### Performance Tests
- Batch indexing 50 videos
- Query latency with large corpus
- Memory usage with vector DB

---

## Migration Strategy

### For Existing Indexes
1. Re-index with enhanced pipeline (optional)
2. Or: Keep existing indexes, add visual embeddings incrementally
3. Backward compatibility: Support both old and new index formats

### Rollout Plan
1. Phase 1: Add mute video support (backward compatible)
2. Phase 2: Add audio query (new feature, doesn't break existing)
3. Phase 3: Add image query (new feature)
4. Phase 4: Optimization and scaling

---

## Expected Outcomes

### ✅ Mute Video Support
- Videos without audio can be indexed
- Visual-only embeddings generated
- Searchable using visual queries

### ✅ Multi-Modal Queries
- Text queries: ✓ (existing)
- Audio queries: ✓ (new)
- Image queries: ✓ (new)
- Video queries: ✓ (existing)

### ✅ Scalability
- Efficient indexing of 50+ videos
- Fast search response times
- Optimized memory usage

---

## Timeline Estimate

- **Phase 1 (Mute Video)**: 2-3 hours
- **Phase 2 (Audio Query)**: 1-2 hours
- **Phase 3 (Image Query)**: 2-3 hours
- **Phase 4 (Unified Router)**: 1-2 hours
- **Phase 5 (Scalability)**: 3-4 hours

**Total**: ~10-14 hours of development time

---

## Next Steps

1. **Review this plan** and confirm requirements
2. **Start with Phase 1** (mute video support) - most critical
3. **Test incrementally** after each phase
4. **Iterate based on feedback**

Would you like me to start implementing these enhancements?


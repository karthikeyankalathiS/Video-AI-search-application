#!/usr/bin/env python3
"""
Flask Web Application for Video Search Agent
Provides a web UI for the AI-powered video search system
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from video_search_agent import VideoSearchAgent
from rag_agent import RAGAgent
import tempfile
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'}
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/highlights', exist_ok=True)
os.makedirs('indexes', exist_ok=True)

# Global agent instances
agent = None
rag_agent = None
current_index_path = None

# Progress tracking for indexing
indexing_progress = {
    'current': 0,
    'total': 0,
    'current_video': '',
    'status': 'idle'  # idle, processing, complete, error
}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_audio_file(filename):
    """Check if audio file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

def allowed_image_file(filename):
    """Check if image file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_agent():
    """Initialize the video search agent."""
    global agent
    try:
        model_size = request.json.get('model_size', 'base')
        use_openai = request.json.get('use_openai', True)  # Use OpenAI by default for better quality
        openai_api_key = request.json.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        
        agent = VideoSearchAgent(
            model_size=model_size,
            use_openai=use_openai,
            openai_api_key=openai_api_key
        )
        return jsonify({'status': 'success', 'message': 'Agent initialized'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/index', methods=['POST'])
def index_videos():
    """Index uploaded videos."""
    global agent, current_index_path, indexing_progress
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if 'videos' not in request.files:
        return jsonify({'status': 'error', 'message': 'No videos uploaded'}), 400
    
    files = request.files.getlist('videos')
    if not files or files[0].filename == '':
        return jsonify({'status': 'error', 'message': 'No files selected'}), 400
    
    try:
        video_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                video_paths.append(filepath)
        
        if not video_paths:
            return jsonify({'status': 'error', 'message': 'No valid video files'}), 400
        
        # Initialize progress tracking
        indexing_progress = {
            'current': 0,
            'total': len(video_paths),
            'current_video': '',
            'status': 'processing'
        }
        
        # Process videos with progress tracking
        for idx, video_path in enumerate(video_paths):
            indexing_progress['current'] = idx + 1
            indexing_progress['current_video'] = os.path.basename(video_path)
            agent.process_video(video_path)
        
        # Save index
        index_filename = request.form.get('index_name', 'default_index.json')
        current_index_path = os.path.join('indexes', index_filename)
        agent.save_index(current_index_path)
        
        # Get statistics
        total_segments = sum(len(v['segments']) for v in agent.video_index)
        
        # Mark as complete
        indexing_progress['status'] = 'complete'
        
        return jsonify({
            'status': 'success',
            'message': f'Indexed {len(video_paths)} video(s)',
            'videos_count': len(video_paths),
            'segments_count': total_segments,
            'index_path': current_index_path
        })
    except Exception as e:
        indexing_progress['status'] = 'error'
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search videos with text query."""
    global agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if not current_index_path or not os.path.exists(current_index_path):
        return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
    
    try:
        data = request.json
        query = data.get('query', '')
        top_k = int(data.get('top_k', 5))
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        # Load index if needed
        if not agent.video_index:
            agent.load_index(current_index_path)
        
        # Search
        results = agent.search_by_text(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                'rank': i,
                'similarity': round(result['similarity'], 3),
                'video_id': result['video_id'],
                'start_time': round(result['start'], 2),
                'end_time': round(result['end'], 2),
                'text': result['segment']['text'],
                'duration': round(result['end'] - result['start'], 2)
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search-clip', methods=['POST'])
def search_clip():
    """Search videos with video clip query."""
    global agent, rag_agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if not current_index_path or not os.path.exists(current_index_path):
        return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
    
    if 'clip' not in request.files:
        return jsonify({'status': 'error', 'message': 'No clip uploaded'}), 400
    
    try:
        file = request.files['clip']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'query_{filename}')
            file.save(filepath)
            
            # Load index if needed
            if not agent.video_index:
                agent.load_index(current_index_path)
            
            top_k = int(request.form.get('top_k', 5))
            
            # Search with clip
            results = agent.search_by_video_clip(filepath, top_k=top_k)
            
            # Additional filtering: remove results with very low similarity
            # This helps prevent irrelevant results from being returned
            min_acceptable_similarity = 0.60  # Minimum acceptable similarity score
            filtered_results = [r for r in results if r['similarity'] >= min_acceptable_similarity]
            
            if not filtered_results and results:
                # If all results were filtered but we had some, return empty with message
                return jsonify({
                    'status': 'success',
                    'message': 'No results found above minimum similarity threshold (0.60). The uploaded clip may not be relevant to the indexed videos.',
                    'results': [],
                    'count': 0
                })
            
            # Format results
            formatted_results = []
            for i, result in enumerate(filtered_results, 1):
                formatted_results.append({
                    'rank': i,
                    'similarity': round(result['similarity'], 3),
                    'video_id': result['video_id'],
                    'start_time': round(result['start'], 2),
                    'end_time': round(result['end'], 2),
                    'text': result['segment'].get('text', ''),
                    'duration': round(result['end'] - result['start'], 2)
                })
            
            # Generate answer using RAG if available
            answer = None
            if rag_agent and results:
                try:
                    # Initialize RAG agent if needed
                    if rag_agent is None:
                        openai_api_key = os.getenv('OPENAI_API_KEY')
                        rag_agent = RAGAgent(
                            video_search_agent=agent,
                            openai_api_key=openai_api_key,
                            model='gpt-3.5-turbo',
                            use_openai=True
                        )
                    # Create a question from the search context
                    question = "What information is in the video segments similar to the uploaded clip?"
                    rag_result = rag_agent.query_with_custom_context(question, results)
                    answer = rag_result.get('answer', '')
                except Exception as e:
                    print(f"  ⚠ RAG answer generation failed: {e}")
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results),
                'answer': answer
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search-audio', methods=['POST'])
def search_audio():
    """Search videos with audio query."""
    global agent, rag_agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if not current_index_path or not os.path.exists(current_index_path):
        return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file uploaded'}), 400
    
    try:
        file = request.files['audio']
        if file and allowed_audio_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'query_audio_{filename}')
            file.save(filepath)
            
            # Load index if needed
            if not agent.video_index:
                agent.load_index(current_index_path)
            
            top_k = int(request.form.get('top_k', 5))
            
            # Search with audio
            results = agent.search_by_audio(filepath, top_k=top_k)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'rank': i,
                    'similarity': round(result['similarity'], 3),
                    'video_id': result['video_id'],
                    'start_time': round(result['start'], 2),
                    'end_time': round(result['end'], 2),
                    'text': result['segment'].get('text', ''),
                    'duration': round(result['end'] - result['start'], 2)
                })
            
            # Generate answer using RAG if available
            answer = None
            if rag_agent and results:
                try:
                    # Initialize RAG agent if needed
                    if rag_agent is None:
                        openai_api_key = os.getenv('OPENAI_API_KEY')
                        rag_agent = RAGAgent(
                            video_search_agent=agent,
                            openai_api_key=openai_api_key,
                            model='gpt-3.5-turbo',
                            use_openai=True
                        )
                    # Transcribe audio to create a question
                    transcript_result = agent.whisper_model.transcribe(filepath)
                    query_text = transcript_result.get('text', '').strip()
                    if query_text:
                        question = f"Based on the video content, what information relates to: {query_text}?"
                    else:
                        question = "What information is in the video segments similar to the uploaded audio?"
                    rag_result = rag_agent.query_with_custom_context(question, results)
                    answer = rag_result.get('answer', '')
                except Exception as e:
                    print(f"  ⚠ RAG answer generation failed: {e}")
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results),
                'answer': answer
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid audio file format'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search-image', methods=['POST'])
def search_image():
    """Search videos with image query."""
    global agent, rag_agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if not current_index_path or not os.path.exists(current_index_path):
        return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file uploaded'}), 400
    
    try:
        file = request.files['image']
        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'query_image_{filename}')
            file.save(filepath)
            
            # Load index if needed
            if not agent.video_index:
                agent.load_index(current_index_path)
            
            top_k = int(request.form.get('top_k', 5))
            
            # Search with image
            results = agent.search_by_image(filepath, top_k=top_k)
            
            # Additional filtering: remove results with very low similarity
            min_acceptable_similarity = 0.65  # Minimum acceptable similarity for image search
            filtered_results = [r for r in results if r['similarity'] >= min_acceptable_similarity]
            
            if not filtered_results and results:
                # If all results were filtered but we had some, return empty with message
                return jsonify({
                    'status': 'success',
                    'message': 'No results found above minimum similarity threshold (0.65). The uploaded image may not be relevant to the indexed videos.',
                    'results': [],
                    'count': 0
                })
            
            # Format results
            formatted_results = []
            for i, result in enumerate(filtered_results, 1):
                formatted_results.append({
                    'rank': i,
                    'similarity': round(result['similarity'], 3),
                    'video_id': result['video_id'],
                    'start_time': round(result['start'], 2),
                    'end_time': round(result['end'], 2),
                    'text': result['segment'].get('text', ''),
                    'duration': round(result['end'] - result['start'], 2)
                })
            
            # Generate answer using RAG if available
            answer = None
            if rag_agent and results:
                try:
                    # Initialize RAG agent if needed
                    if rag_agent is None:
                        openai_api_key = os.getenv('OPENAI_API_KEY')
                        rag_agent = RAGAgent(
                            video_search_agent=agent,
                            openai_api_key=openai_api_key,
                            model='gpt-3.5-turbo',
                            use_openai=True
                        )
                    # Create a question from the image search context
                    question = "What information is in the video segments that are visually similar to the uploaded image?"
                    rag_result = rag_agent.query_with_custom_context(question, results)
                    answer = rag_result.get('answer', '')
                except Exception as e:
                    print(f"  ⚠ RAG answer generation failed: {e}")
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results),
                'answer': answer
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid image file format'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate-highlight', methods=['POST'])
def generate_highlight():
    """Generate highlight reel from search results."""
    global agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    try:
        data = request.json
        results_data = data.get('results', [])
        max_duration = int(data.get('max_duration', 60))
        
        if not results_data:
            return jsonify({'status': 'error', 'message': 'No results provided'}), 400
        
        # Load index if needed
        if not agent.video_index:
            if not current_index_path or not os.path.exists(current_index_path):
                return jsonify({'status': 'error', 'message': 'No index found'}), 400
            agent.load_index(current_index_path)
        
        # Reconstruct results format
        search_results = []
        for r in results_data:
            # Find matching segment
            for video_data in agent.video_index:
                if video_data['video_id'] == r['video_id']:
                    for segment in video_data['segments']:
                        if abs(segment['start'] - r['start_time']) < 0.1:
                            search_results.append({
                                'video_path': video_data['video_path'],
                                'start': r['start_time'],
                                'end': r['end_time'],
                                'segment': segment
                            })
                            break
                    break
        
        if not search_results:
            return jsonify({'status': 'error', 'message': 'Could not find segments'}), 400
        
        # Generate highlight
        highlight_filename = f'highlight_{int(os.urandom(4).hex(), 16)}.mp4'
        highlight_path = os.path.join('static/highlights', highlight_filename)
        
        agent.generate_highlight_reel(search_results, highlight_path, max_duration)
        
        return jsonify({
            'status': 'success',
            'highlight_url': f'/static/highlights/{highlight_filename}',
            'filename': highlight_filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/preview-segment', methods=['POST'])
def preview_segment():
    """Generate a preview video clip for a specific segment."""
    global agent, current_index_path
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if not current_index_path or not os.path.exists(current_index_path):
        return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
    
    try:
        data = request.json
        video_id = data.get('video_id')
        start_time = float(data.get('start_time'))
        end_time = float(data.get('end_time'))
        
        if not video_id:
            return jsonify({'status': 'error', 'message': 'video_id is required'}), 400
        
        # Load index if needed
        if not agent.video_index:
            agent.load_index(current_index_path)
        
        # Find video path
        video_path = None
        for video_data in agent.video_index:
            if video_data['video_id'] == video_id:
                video_path = video_data['video_path']
                break
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'status': 'error', 'message': f'Video file not found: {video_id}'}), 400
        
        # Generate preview clip
        from moviepy.editor import VideoFileClip
        
        video = VideoFileClip(video_path)
        max_end = min(end_time, video.duration)
        clip = video.subclip(start_time, max_end)
        
        # Save preview
        preview_filename = f'preview_{int(os.urandom(4).hex(), 16)}.mp4'
        preview_path = os.path.join('static/highlights', preview_filename)
        
        try:
            clip.write_videofile(
                preview_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
        except Exception as e:
            # Fallback without audio
            clip_no_audio = clip.without_audio() if clip.audio else clip
            clip_no_audio.write_videofile(
                preview_path,
                codec='libx264',
                audio=False,
                verbose=False,
                logger=None
            )
            clip_no_audio.close()
        
        clip.close()
        video.close()
        
        return jsonify({
            'status': 'success',
            'preview_url': f'/static/highlights/{preview_filename}',
            'filename': preview_filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/rag-query', methods=['POST'])
def rag_query():
    """Answer questions using RAG (Retrieval-Augmented Generation)."""
    global agent, rag_agent, current_index_path
    
    try:
        if agent is None:
            return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
        
        if not current_index_path or not os.path.exists(current_index_path):
            return jsonify({'status': 'error', 'message': 'No index found. Please index videos first.'}), 400
        
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({'status': 'error', 'message': 'Invalid JSON in request'}), 400
        
        question = data.get('question', '')
        top_k = int(data.get('top_k', 5))
        use_openai = data.get('use_openai', True)
        openai_api_key = data.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        model = data.get('model', 'gpt-3.5-turbo')
        
        if not question:
            return jsonify({'status': 'error', 'message': 'Question is required'}), 400
        
        # Load index if needed
        if not agent.video_index:
            agent.load_index(current_index_path)
        
        # Initialize or update RAG agent if needed
        try:
            if rag_agent is None or (use_openai and not rag_agent.use_openai):
                rag_agent = RAGAgent(
                    video_search_agent=agent,
                    openai_api_key=openai_api_key,
                    model=model,
                    use_openai=use_openai
                )
            elif use_openai and rag_agent.model != model:
                # Update model if changed
                rag_agent.model = model
        except Exception as rag_init_error:
            return jsonify({'status': 'error', 'message': f'Failed to initialize RAG agent: {str(rag_init_error)}'}), 500
        
        # Query with RAG
        result = rag_agent.query(question, top_k=top_k)
        
        # Format response
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'retrieval_count': result['retrieval_count']
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"RAG Query Error: {error_trace}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of the agent."""
    global agent, rag_agent, current_index_path
    
    status = {
        'agent_initialized': agent is not None,
        'rag_agent_initialized': rag_agent is not None,
        'index_loaded': False,
        'videos_count': 0,
        'segments_count': 0
    }
    
    if agent and agent.video_index:
        status['index_loaded'] = True
        status['videos_count'] = len(agent.video_index)
        status['segments_count'] = sum(len(v['segments']) for v in agent.video_index)
    
    return jsonify(status)

@app.route('/api/indexing-progress', methods=['GET'])
def get_indexing_progress():
    """Get current indexing progress."""
    global indexing_progress
    progress_percent = 0
    if indexing_progress['total'] > 0:
        progress_percent = int((indexing_progress['current'] / indexing_progress['total']) * 100)
    
    return jsonify({
        'progress': progress_percent,
        'current': indexing_progress['current'],
        'total': indexing_progress['total'],
        'current_video': indexing_progress['current_video'],
        'status': indexing_progress['status']
    })

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get list of all indexed videos."""
    global agent, current_index_path
    
    # Try to load index if agent exists but index is empty
    if agent is not None and (not agent.video_index or len(agent.video_index) == 0):
        # Try to load from default index if available
        default_index = os.path.join('indexes', 'default_index.json')
        if current_index_path and os.path.exists(current_index_path):
            try:
                agent.load_index(current_index_path)
            except Exception as e:
                print(f"Error loading index: {e}")
        elif os.path.exists(default_index):
            try:
                agent.load_index(default_index)
                current_index_path = default_index
            except Exception as e:
                print(f"Error loading default index: {e}")
    
    if agent is None:
        return jsonify({'status': 'success', 'videos': [], 'message': 'Agent not initialized'})
    
    if not agent.video_index or len(agent.video_index) == 0:
        return jsonify({'status': 'success', 'videos': [], 'message': 'No videos indexed yet'})
    
    videos = []
    for video_data in agent.video_index:
        videos.append({
            'video_id': video_data.get('video_id', 'Unknown'),
            'video_path': video_data.get('video_path', ''),
            'filename': os.path.basename(video_data.get('video_path', '')),
            'duration': video_data.get('duration', 0),
            'segments_count': len(video_data.get('segments', [])),
            'language': video_data.get('language', 'Unknown'),
            'has_audio': video_data.get('has_audio', False)
        })
    
    return jsonify({
        'status': 'success',
        'videos': videos,
        'total': len(videos)
    })

@app.route('/api/video/<path:filename>', methods=['GET'])
def serve_video(filename):
    """Serve video file for playback."""
    try:
        # Security: ensure filename doesn't contain path traversal
        safe_filename = secure_filename(os.path.basename(filename))
        
        # First try the uploads folder with the safe filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        # If not found, check if the video_path from index is a full path
        if not os.path.exists(video_path):
            # Try to find the video by checking the index
            global agent
            if agent and agent.video_index:
                for video_data in agent.video_index:
                    stored_path = video_data.get('video_path', '')
                    stored_filename = os.path.basename(stored_path)
                    if stored_filename == safe_filename or stored_filename == filename:
                        # Use the stored path if it exists
                        if os.path.exists(stored_path):
                            video_path = stored_path
                            break
                        # Otherwise try uploads folder with stored filename
                        elif os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)):
                            video_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
                            break
        
        if not os.path.exists(video_path):
            return jsonify({'status': 'error', 'message': f'Video file not found: {safe_filename}'}), 404
        
        # Determine MIME type based on file extension
        ext = os.path.splitext(video_path)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        mimetype = mime_types.get(ext, 'video/mp4')
        
        return send_file(video_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Initialize agent on startup with OpenAI embeddings for better quality
    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("⚠ Warning: OPENAI_API_KEY not set. OpenAI embeddings will not be available.")
            openai_api_key = None
        agent = VideoSearchAgent(model_size='base', use_openai=True, openai_api_key=openai_api_key)
        print("✓ Video Search Agent initialized with OpenAI embeddings")
        
        # Try to load default index if it exists
        default_index = os.path.join('indexes', 'default_index.json')
        if os.path.exists(default_index):
            try:
                agent.load_index(default_index)
                current_index_path = default_index
                print(f"✓ Loaded default index: {len(agent.video_index)} videos")
            except Exception as e:
                print(f"⚠ Could not load default index: {e}")
        
        # Initialize RAG agent
        try:
            rag_agent = RAGAgent(
                video_search_agent=agent,
                openai_api_key=openai_api_key,
                model='gpt-3.5-turbo',
                use_openai=True
            )
        except Exception as rag_e:
            print(f"⚠ Warning: Could not initialize RAG agent: {rag_e}")
            rag_agent = None
    except Exception as e:
        print(f"⚠ Warning: Could not initialize agent with OpenAI: {e}")
        print("   Falling back to sentence-transformers...")
        try:
            agent = VideoSearchAgent(model_size='base', use_openai=False)
            print("✓ Video Search Agent initialized (using sentence-transformers)")
            # Initialize RAG agent without OpenAI
            rag_agent = RAGAgent(
                video_search_agent=agent,
                openai_api_key=None,
                model='gpt-3.5-turbo',
                use_openai=False
            )
        except Exception as e2:
            print(f"⚠ Warning: Could not initialize agent: {e2}")
            print("   Agent will be initialized on first use")
            rag_agent = None
    
    print("\n" + "="*70)
    print("🚀 Starting Video Search Agent Web Server")
    print("="*70)
    print("📍 Server running at: http://localhost:5001")
    print("📍 Or access via: http://127.0.0.1:5001")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)

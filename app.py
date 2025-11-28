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

# Global agent instance
agent = None
current_index_path = None

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
    global agent, current_index_path
    
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
        
        # Process videos
        agent.process_corpus(video_paths)
        
        # Save index
        index_filename = request.form.get('index_name', 'default_index.json')
        current_index_path = os.path.join('indexes', index_filename)
        agent.save_index(current_index_path)
        
        # Get statistics
        total_segments = sum(len(v['segments']) for v in agent.video_index)
        
        return jsonify({
            'status': 'success',
            'message': f'Indexed {len(video_paths)} video(s)',
            'videos_count': len(video_paths),
            'segments_count': total_segments,
            'index_path': current_index_path
        })
    except Exception as e:
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
    global agent, current_index_path
    
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
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search-audio', methods=['POST'])
def search_audio():
    """Search videos with audio query."""
    global agent, current_index_path
    
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
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results)
            })
        else:
            return jsonify({'status': 'error', 'message': 'Invalid audio file format'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search-image', methods=['POST'])
def search_image():
    """Search videos with image query."""
    global agent, current_index_path
    
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
            
            # Clean up query file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'results': formatted_results,
                'count': len(formatted_results)
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

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of the agent."""
    global agent, current_index_path
    
    status = {
        'agent_initialized': agent is not None,
        'index_loaded': False,
        'videos_count': 0,
        'segments_count': 0
    }
    
    if agent and agent.video_index:
        status['index_loaded'] = True
        status['videos_count'] = len(agent.video_index)
        status['segments_count'] = sum(len(v['segments']) for v in agent.video_index)
    
    return jsonify(status)

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
    except Exception as e:
        print(f"⚠ Warning: Could not initialize agent with OpenAI: {e}")
        print("   Falling back to sentence-transformers...")
        try:
            agent = VideoSearchAgent(model_size='base', use_openai=False)
            print("✓ Video Search Agent initialized (using sentence-transformers)")
        except Exception as e2:
            print(f"⚠ Warning: Could not initialize agent: {e2}")
            print("   Agent will be initialized on first use")
    
    print("\n" + "="*70)
    print("🚀 Starting Video Search Agent Web Server")
    print("="*70)
    print("📍 Server running at: http://localhost:5001")
    print("📍 Or access via: http://127.0.0.1:5001")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)

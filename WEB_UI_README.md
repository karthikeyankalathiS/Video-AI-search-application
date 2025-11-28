# Web UI for Video Search Agent

A beautiful, modern web interface for the AI-Powered Video Search Agent.

## Features

✅ **Modern Web Interface** - Clean, responsive design  
✅ **Video Indexing** - Upload and index multiple videos  
✅ **Text Search** - Natural language query search  
✅ **Clip Search** - Upload video clip to find similar segments  
✅ **Results Display** - Ranked results with similarity scores  
✅ **Highlight Reel Generation** - Automatic compilation of top segments  
✅ **Index Management** - Load and manage multiple indexes  

## Installation

```bash
# Install Flask and dependencies
pip install flask werkzeug

# Or install all requirements
pip install -r requirements.txt
```

## Running the Web UI

```bash
# Activate virtual environment
source venv/bin/activate

# Run the Flask app
python3 app.py
```

The server will start on `http://localhost:5000`

Open your browser and navigate to the URL to access the web interface.

## Usage

### 1. Index Videos
- Go to "Index Videos" tab
- Select one or more video files
- Enter an index name (optional)
- Click "Index Videos"
- Wait for processing to complete

### 2. Text Search
- Go to "Text Search" tab
- Enter your query (e.g., "Spring Framework", "dependency injection")
- Set number of results (default: 5)
- Click "Search"
- View results in "Results" tab

### 3. Clip Search
- Go to "Clip Search" tab
- Upload a sample video clip
- Set number of results
- Click "Search with Clip"
- View similar segments in "Results" tab

### 4. Generate Highlight Reel
- After getting search results
- Go to "Results" tab
- Set maximum duration (default: 60 seconds)
- Click "Generate Highlight Reel"
- Download the generated video

## Project Structure

```
videoAi/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css    # Styling
│   └── js/
│       └── app.js        # Frontend JavaScript
├── uploads/              # Uploaded videos (auto-created)
├── indexes/              # Saved indexes (auto-created)
└── highlights/           # Generated highlight reels (auto-created)
```

## API Endpoints

- `POST /api/index` - Index video files
- `POST /api/load_index` - Load existing index
- `POST /api/search` - Text query search
- `POST /api/search_clip` - Video clip search
- `POST /api/generate_highlight` - Generate highlight reel
- `GET /api/list_indexes` - List available indexes
- `GET /api/download_highlight/<filename>` - Download highlight

## Features

### Modern UI
- Gradient design
- Responsive layout
- Tab-based navigation
- Real-time status updates
- Interactive results display

### Functionality
- Multiple video upload
- Index management
- Real-time search
- Similarity scoring
- Highlight generation
- File downloads

## Troubleshooting

**Port already in use:**
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Upload fails:**
- Check file size (max 500MB)
- Ensure video format is supported
- Check uploads/ directory permissions

**Index not loading:**
- Ensure index file exists in indexes/ directory
- Check file permissions
- Verify JSON format

## Demo Workflow

1. Start server: `python3 app.py`
2. Open browser: `http://localhost:5000`
3. Index video: Upload video.mp4
4. Search: Query "Spring Framework"
5. View results: See ranked segments
6. Generate highlight: Create summary video
7. Download: Get the highlight reel

Enjoy your AI-powered video search interface! 🎬


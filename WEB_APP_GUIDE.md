# Web UI Guide - Video Search Agent

## Overview

A modern web interface for the AI-Powered Video Search Agent with:
- **Video Indexing**: Upload and process videos
- **Text Search**: Natural language query search
- **Clip Search**: Upload a video clip to find similar segments
- **Highlight Reels**: Automatically generate summary videos

## Installation

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Flask and dependencies
pip install flask werkzeug

# Or install all requirements
pip install -r requirements.txt
```

### 2. Create Required Directories

The app will create these automatically, but you can create them manually:

```bash
mkdir -p uploads static/highlights indexes templates static/css static/js
```

## Running the Web App

### Start the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the Flask app
python3 app.py
```

The app will start on: **http://localhost:5000**

Open your browser and navigate to: `http://localhost:5000`

## Usage

### Step 1: Index Videos

1. Click on the **"Index Videos"** tab
2. Select one or more video files (MP4, MOV, AVI, MKV)
3. Choose an index name (optional, default: `default_index.json`)
4. Select Whisper model size (Base recommended)
5. Click **"Index Videos"**
6. Wait for processing to complete (2-5 minutes per video)

### Step 2: Text Search

1. Click on the **"Text Search"** tab
2. Enter your search query (e.g., "Spring Framework", "dependency injection")
3. Set number of results (default: 5)
4. Click **"Search"**
5. View ranked results with similarity scores
6. Click **"Generate Highlight Reel"** to create a summary video

### Step 3: Clip Search

1. Click on the **"Clip Search"** tab
2. Upload a sample video clip
3. Set number of results
4. Click **"Search with Clip"**
5. View similar segments found
6. Generate highlight reel if desired

## Features

### Modern UI
- Clean, responsive design
- Tab-based navigation
- Real-time status updates
- Smooth animations

### Search Capabilities
- **Text Queries**: Natural language search
- **Clip Queries**: Find similar segments using video
- **Ranked Results**: Sorted by semantic similarity
- **Detailed Info**: Timestamps, similarity scores, transcript text

### Highlight Reels
- Automatic compilation of top segments
- Configurable maximum duration
- Video player with controls
- Downloadable output

## API Endpoints

The web app uses these REST API endpoints:

- `GET /` - Main page
- `POST /api/initialize` - Initialize agent
- `POST /api/index` - Index videos
- `POST /api/search` - Text search
- `POST /api/search-clip` - Clip search
- `POST /api/generate-highlight` - Generate highlight reel
- `GET /api/status` - Get agent status

## File Structure

```
videoAi/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css    # Stylesheet
│   ├── js/
│   │   └── app.js       # JavaScript
│   └── highlights/      # Generated highlight reels
├── uploads/             # Uploaded videos
└── indexes/             # Saved indexes
```

## Troubleshooting

### Port Already in Use

If port 5000 is busy, modify `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### File Upload Size

Default max size is 500MB. To change:

```python
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB
```

### CORS Issues

If accessing from different domain, add CORS support:

```bash
pip install flask-cors
```

Then in `app.py`:
```python
from flask_cors import CORS
CORS(app)
```

## Production Deployment

For production use:

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Disable debug mode**:
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```

3. **Set up reverse proxy** (nginx/Apache)

4. **Use environment variables** for configuration

## Screenshots

The UI includes:
- Gradient purple theme
- Card-based layout
- Status indicators
- Result cards with similarity scores
- Video player for highlights

## Next Steps

1. **Add user authentication** (optional)
2. **Add progress bars** for long operations
3. **Add video preview** in results
4. **Add export options** for results
5. **Add batch processing** for multiple queries


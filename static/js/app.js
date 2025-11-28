// Global state
let currentResults = [];
let currentClipResults = [];
let currentAudioResults = [];
let currentImageResults = [];
let currentRAGResults = [];  // Store RAG results for highlight generation

// Separate selection tracking for each tab
let selectedResults = {
    text: new Set(),
    clip: new Set(),
    audio: new Set(),
    image: new Set(),
    rag: new Set()  // Add RAG selection tracking
};

// Clear selections for a specific type (define early so it's available to all handlers)
function clearSelectionsForType(type) {
    if (selectedResults[type]) {
        selectedResults[type].clear();
    }
}

// ============================================
// Toast Notification System
// ============================================

function showToast(message, type = 'info', title = '') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type]} toast-icon"></i>
        <div class="toast-content">
            ${title ? `<div class="toast-title">${title}</div>` : ''}
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.animation = 'slideInRight 0.3s ease-out reverse';
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

// ============================================
// Loading Overlay System
// ============================================

function showLoadingOverlay(message = 'Processing your request...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = overlay?.querySelector('.loading-text');
    if (overlay && loadingText) {
        loadingText.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function updateProgress(percent) {
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }
}

// ============================================
// Drag & Drop File Upload
// ============================================

function setupDragAndDrop() {
    // Video upload area
    const videoUploadArea = document.getElementById('videoUploadArea');
    const videoInput = document.getElementById('videos');
    if (videoUploadArea && videoInput) {
        setupFileUploadArea(videoUploadArea, videoInput, 'video', true);
    }
    
    // Clip upload area
    const clipUploadArea = document.getElementById('clipUploadArea');
    const clipInput = document.getElementById('clip');
    if (clipUploadArea && clipInput) {
        setupFileUploadArea(clipUploadArea, clipInput, 'video', false);
    }
    
    // Audio upload area
    const audioUploadArea = document.getElementById('audioUploadArea');
    const audioInput = document.getElementById('audio');
    if (audioUploadArea && audioInput) {
        setupFileUploadArea(audioUploadArea, audioInput, 'audio', false);
    }
    
    // Image upload area
    const imageUploadArea = document.getElementById('imageUploadArea');
    const imageInput = document.getElementById('image');
    if (imageUploadArea && imageInput) {
        setupFileUploadArea(imageUploadArea, imageInput, 'image', false);
    }
}

function setupFileUploadArea(area, input, type, multiple) {
    const fileList = area.querySelector('.file-list') || area.querySelector('.file-preview');
    
    // Flag to prevent double-triggering
    let isOpeningDialog = false;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        area.addEventListener(eventName, () => {
            area.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        area.addEventListener(eventName, () => {
            area.classList.remove('dragover');
        }, false);
    });
    
    area.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        input.files = files;
        handleFileSelection(input, files, type, multiple);
    }, false);
    
    // Prevent input from triggering area click
    input.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    // Handle area click - open file dialog
    area.addEventListener('click', (e) => {
        // Don't trigger if clicking directly on the input
        if (e.target === input) {
            return;
        }
        
        // Prevent double-triggering
        if (isOpeningDialog) {
            return;
        }
        
        isOpeningDialog = true;
        e.preventDefault();
        e.stopPropagation();
        
        // Reset flag after a short delay
        setTimeout(() => {
            isOpeningDialog = false;
        }, 300);
        
        input.click();
    });
    
    input.addEventListener('change', (e) => {
        handleFileSelection(input, e.target.files, type, multiple);
    });
}

function handleFileSelection(input, files, type, multiple) {
    const area = input.closest('.file-upload-area');
    const fileList = area.querySelector('.file-list') || area.querySelector('.file-preview');
    
    if (!files || files.length === 0) return;
    
    if (multiple) {
        // Multiple files (video indexing)
        const fileListDiv = area.querySelector('.file-list');
        if (!fileListDiv) return;
        
        fileListDiv.innerHTML = '';
        Array.from(files).forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-item-info">
                    <i class="fas fa-file-video"></i>
                    <div>
                        <div class="file-item-name">${file.name}</div>
                        <div class="file-item-size">${formatFileSize(file.size)}</div>
                    </div>
                </div>
            `;
            fileListDiv.appendChild(fileItem);
        });
    } else {
        // Single file (preview)
        const previewDiv = area.querySelector('.file-preview');
        if (!previewDiv) return;
        
        previewDiv.innerHTML = '';
        previewDiv.classList.add('active');
        
        const file = files[0];
        if (type === 'image' && file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.onload = () => URL.revokeObjectURL(img.src);
            previewDiv.appendChild(img);
        } else if (type === 'video' && file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.controls = true;
            video.onloadedmetadata = () => URL.revokeObjectURL(video.src);
            previewDiv.appendChild(video);
        } else if (type === 'audio' && file.type.startsWith('audio/')) {
            const audio = document.createElement('audio');
            audio.src = URL.createObjectURL(file);
            audio.controls = true;
            audio.onloadedmetadata = () => URL.revokeObjectURL(audio.src);
            previewDiv.appendChild(audio);
        }
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ============================================
// Enhanced Status Updates
// ============================================

function updateHeaderStats(videosCount, segmentsCount) {
    const videoCountEl = document.getElementById('videoCount');
    const segmentCountEl = document.getElementById('segmentCount');
    
    if (videoCountEl) videoCountEl.textContent = videosCount || 0;
    if (segmentCountEl) segmentCountEl.textContent = segmentsCount || 0;
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupEventListeners();
    setupDragAndDrop();
    loadVideosList(); // Load videos list on page load
});

// Check agent status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.agent_initialized) {
            updateStatus('System ready', 'success');
            if (data.index_loaded) {
                updateStatus(`Index loaded: ${data.videos_count} video(s), ${data.segments_count} segments`, 'info');
                updateHeaderStats(data.videos_count, data.segments_count);
                showToast(`Index loaded with ${data.videos_count} videos and ${data.segments_count} segments`, 'success', 'System Ready');
            } else {
                updateStatus('System ready. Please index videos to start searching.', 'info');
                updateHeaderStats(0, 0);
            }
        } else {
            updateStatus('Initializing system...', 'info');
            // Initialize agent
            await fetch('/api/initialize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_size: 'base'})
            });
            updateStatus('System initialized. Please index videos.', 'success');
            showToast('System initialized successfully', 'success', 'Initialization Complete');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        showToast('Failed to initialize system: ' + error.message, 'error', 'Initialization Error');
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('indexForm').addEventListener('submit', handleIndex);
    document.getElementById('ragForm').addEventListener('submit', handleRAGQuery);
    document.getElementById('clipSearchForm').addEventListener('submit', handleClipSearch);
    document.getElementById('audioSearchForm').addEventListener('submit', handleAudioSearch);
    document.getElementById('imageSearchForm').addEventListener('submit', handleImageSearch);
    document.getElementById('generateClipHighlightBtn').addEventListener('click', generateClipHighlight);
    document.getElementById('generateAudioHighlightBtn').addEventListener('click', generateAudioHighlight);
    document.getElementById('generateImageHighlightBtn').addEventListener('click', generateImageHighlight);
    
    // RAG highlight button
    const ragHighlightBtn = document.getElementById('generateRAGHighlightBtn');
    if (ragHighlightBtn) {
        ragHighlightBtn.addEventListener('click', generateRAGHighlight);
    }
    
    // Toggle API key field visibility
    const ragUseOpenAI = document.getElementById('ragUseOpenAI');
    const ragApiKeyGroup = document.getElementById('ragApiKeyGroup');
    if (ragUseOpenAI && ragApiKeyGroup) {
        ragUseOpenAI.addEventListener('change', function() {
            ragApiKeyGroup.style.display = this.checked ? 'block' : 'none';
        });
    }
}

// Tab switching
function showTab(tabName, buttonElement) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    const tabElement = document.getElementById(tabName + 'Tab');
    if (tabElement) {
        tabElement.classList.add('active');
    }
    
    // Activate button
    if (buttonElement) {
        buttonElement.classList.add('active');
    } else if (window.event && window.event.target) {
        window.event.target.classList.add('active');
    } else {
        // Fallback: find button by text content
        document.querySelectorAll('.tab-btn').forEach(btn => {
            if (btn.textContent.trim().includes(tabName.replace('-', ' '))) {
                btn.classList.add('active');
            }
        });
    }
    
    // Update selection count for the active tab
    const typeMap = {
        'clip-search': 'clip',
        'audio-search': 'audio',
        'image-search': 'image'
    };
    const resultsType = typeMap[tabName];
    if (resultsType) {
        updateSelectionCountForType(resultsType);
    }
}

// Update status bar
function updateStatus(message, type = 'info') {
    const statusBar = document.getElementById('statusBar');
    const statusText = document.getElementById('statusText');
    
    if (statusBar && statusText) {
        statusText.textContent = message;
        statusBar.className = 'status-bar ' + type;
        
        // Update icon based on type
        const icon = statusBar.querySelector('.status-icon');
        if (icon) {
            const icons = {
                success: 'fa-check-circle',
                error: 'fa-exclamation-circle',
                warning: 'fa-exclamation-triangle',
                info: 'fa-circle-notch fa-spin'
            };
            icon.className = `fas ${icons[type] || icons.info} status-icon`;
        }
    }
}

// Handle video indexing
async function handleIndex(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const btn = document.getElementById('indexBtn');
    const btnText = document.getElementById('indexBtnText');
    const spinner = document.getElementById('indexSpinner');
    const resultBox = document.getElementById('indexResult');
    const progressDiv = document.getElementById('indexingProgress');
    const progressBar = document.getElementById('indexingProgressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progressText');
    
    btn.disabled = true;
    btnText.textContent = 'Indexing...';
    spinner.style.display = 'inline-block';
    resultBox.style.display = 'none';
    progressDiv.style.display = 'block';
    updateStatus('Indexing videos... This may take a few minutes.', 'info');
    
    // Reset progress
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressText.textContent = 'Preparing...';
    
    // Start progress polling
    const progressInterval = setInterval(async () => {
        try {
            const progressResponse = await fetch('/api/indexing-progress');
            const progressData = await progressResponse.json();
            
            if (progressData.status === 'processing') {
                const percent = progressData.progress || 0;
                progressBar.style.width = `${percent}%`;
                progressPercent.textContent = `${percent}%`;
                if (progressData.total > 0) {
                    progressText.textContent = `Processing ${progressData.current} of ${progressData.total} videos: ${progressData.current_video || 'Preparing...'}`;
                } else {
                    progressText.textContent = progressData.current_video || 'Preparing...';
                }
            } else if (progressData.status === 'complete' || progressData.status === 'error') {
                clearInterval(progressInterval);
                if (progressData.status === 'complete') {
                    progressBar.style.width = '100%';
                    progressPercent.textContent = '100%';
                    progressText.textContent = 'Indexing complete!';
                }
            }
        } catch (err) {
            console.error('Error fetching progress:', err);
        }
    }, 500); // Poll every 500ms
    
    try {
        const response = await fetch('/api/index', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Clear progress polling
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        
        if (data.status === 'success') {
            resultBox.style.display = 'block';
            resultBox.innerHTML = `
                <h4><i class="fas fa-check-circle"></i> Indexing Complete!</h4>
                <p><strong>Videos:</strong> ${data.videos_count}</p>
                <p><strong>Segments:</strong> ${data.segments_count}</p>
                <p><strong>Index saved:</strong> ${data.index_path}</p>
            `;
            resultBox.style.borderLeftColor = '#28a745';
            updateStatus(`Indexed ${data.videos_count} video(s) with ${data.segments_count} segments`, 'success');
            updateHeaderStats(data.videos_count, data.segments_count);
            showToast(`Successfully indexed ${data.videos_count} videos with ${data.segments_count} segments`, 'success', 'Indexing Complete');
            
            // Refresh videos list
            loadVideosList();
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressDiv.style.display = 'none';
        resultBox.style.display = 'block';
        resultBox.innerHTML = `<p style="color: red;"><i class="fas fa-exclamation-circle"></i> Error: ${error.message}</p>`;
        resultBox.style.borderLeftColor = '#dc3545';
        updateStatus('Error: ' + error.message, 'error');
        showToast('Indexing failed: ' + error.message, 'error', 'Indexing Error');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Index Videos';
        spinner.style.display = 'none';
    }
}

// Load and display videos list
async function loadVideosList() {
    const videosList = document.getElementById('videosList');
    if (!videosList) return;
    
    videosList.innerHTML = '<div class="loading-placeholder"><i class="fas fa-spinner fa-spin"></i> Loading videos...</div>';
    
    try {
        const response = await fetch('/api/videos');
        const data = await response.json();
        
        if (data.status === 'success') {
            if (data.videos && data.videos.length > 0) {
                videosList.innerHTML = '';
                data.videos.forEach((video, index) => {
                    const videoCard = document.createElement('div');
                    videoCard.className = 'video-card';
                    // Use filename for the URL - the backend will handle finding the file
                    const videoUrl = `/api/video/${encodeURIComponent(video.filename)}`;
                    videoCard.innerHTML = `
                        <div class="video-card-header">
                            <div class="video-number">#${index + 1}</div>
                            <div class="video-icon">
                                <i class="fas fa-video"></i>
                            </div>
                        </div>
                        <div class="video-card-body">
                            <h4 class="video-title" title="${video.filename}">${video.filename}</h4>
                            <div class="video-meta">
                                <div class="video-meta-item">
                                    <i class="fas fa-clock"></i>
                                    <span>${formatDuration(video.duration)}</span>
                                </div>
                                <div class="video-meta-item">
                                    <i class="fas fa-film"></i>
                                    <span>${video.segments_count} segments</span>
                                </div>
                                <div class="video-meta-item">
                                    <i class="fas fa-language"></i>
                                    <span>${video.language || 'Unknown'}</span>
                                </div>
                                <div class="video-meta-item">
                                    <i class="fas ${video.has_audio ? 'fa-volume-up' : 'fa-volume-mute'}"></i>
                                    <span>${video.has_audio ? 'With Audio' : 'Mute'}</span>
                                </div>
                            </div>
                            <div class="video-card-actions">
                                <button class="btn btn-primary btn-play" onclick="playVideo('${videoUrl.replace(/'/g, "\\'")}', '${video.filename.replace(/'/g, "\\'")}')">
                                    <i class="fas fa-play"></i>
                                    Play Video
                                </button>
                            </div>
                        </div>
                    `;
                    videosList.appendChild(videoCard);
                });
            } else {
                videosList.innerHTML = `
                    <div class="no-videos">
                        <i class="fas fa-inbox"></i>
                        <p>No videos indexed yet</p>
                        <p class="hint">Upload and index videos to see them here</p>
                    </div>
                `;
            }
        } else {
            throw new Error(data.message || 'Failed to load videos');
        }
    } catch (error) {
        videosList.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error loading videos: ${error.message}</p>
            </div>
        `;
    }
}

// Format duration in seconds to readable format
function formatDuration(seconds) {
    if (!seconds || seconds === 0) return '0s';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Handle text search
async function handleSearch(e) {
    e.preventDefault();
    
    // Clear previous selections for this search type
    if (typeof clearSelectionsForType === 'function') {
        clearSelectionsForType('text');
    }
    
    const query = document.getElementById('query').value;
    const topK = document.getElementById('topK').value;
    const btn = document.getElementById('searchBtn');
    const btnText = document.getElementById('searchBtnText');
    const spinner = document.getElementById('searchSpinner');
    const resultsContainer = document.getElementById('searchResults');
    const resultsList = document.getElementById('resultsList');
    
    btn.disabled = true;
    btnText.textContent = 'Searching...';
    spinner.style.display = 'inline-block';
    resultsContainer.style.display = 'none';
    updateStatus('Searching...', 'info');
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query, top_k: topK})
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentResults = data.results || [];
            
            // Always show results container (even if empty, to show "No Data Found" message)
            resultsContainer.style.display = 'block';
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentResults, resultsList, 'text');
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateHighlightBtn');
            if (data.count > 0 && currentResults.length > 0) {
                highlightBtn.style.display = 'block';
                highlightBtn.textContent = 'Generate Highlight Reel (Select segments above)';
                updateStatus(`Found ${data.count} relevant segments`, 'success');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No relevant segments found', 'info');
            }
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Search error: ' + error.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Search';
        spinner.style.display = 'none';
    }
}

// Handle RAG query
async function handleRAGQuery(e) {
    e.preventDefault();
    
    const question = document.getElementById('ragQuestion').value;
    const topK = document.getElementById('ragTopK').value;
    const model = document.getElementById('ragModel').value;
    const useOpenAI = document.getElementById('ragUseOpenAI').checked;
    const apiKey = document.getElementById('ragApiKey').value;
    
    const btn = document.getElementById('ragBtn');
    const btnText = document.getElementById('ragBtnText');
    const spinner = document.getElementById('ragSpinner');
    const resultsContainer = document.getElementById('ragResults');
    const answerBox = document.getElementById('ragAnswer');
    const sourcesList = document.getElementById('ragSourcesList');
    
    btn.disabled = true;
    btnText.textContent = 'Generating answer...';
    spinner.style.display = 'inline-block';
    resultsContainer.style.display = 'none';
    updateStatus('Retrieving context and generating answer...', 'info');
    
    try {
        const response = await fetch('/api/rag-query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question: question,
                top_k: parseInt(topK),
                model: model,
                use_openai: useOpenAI,
                openai_api_key: apiKey || undefined
            })
        });
        
        // Check if response is OK and is JSON
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText.substring(0, 200)}`);
        }
        
        // Check content type
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const errorText = await response.text();
            throw new Error(`Expected JSON but got ${contentType}. Response: ${errorText.substring(0, 200)}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Store RAG results globally for highlight generation
            currentRAGResults = data.sources || [];
            
            // Display answer
            answerBox.innerHTML = `<div class="rag-answer-content">${formatRAGAnswer(data.answer)}</div>`;
            
            // Display sources with preview and selection
            if (data.sources && data.sources.length > 0) {
                displayRAGSources(data.sources, sourcesList);
                
                // Update results count
                const resultsCountEl = document.getElementById('ragResultsCount');
                if (resultsCountEl) {
                    resultsCountEl.textContent = `(${data.retrieval_count || data.sources.length} found)`;
                }
                
                // Show highlight button
                const highlightBtn = document.getElementById('generateRAGHighlightBtn');
                if (highlightBtn) {
                    highlightBtn.style.display = 'block';
                }
            } else {
                sourcesList.innerHTML = '<p>No source segments found.</p>';
            }
            
            resultsContainer.style.display = 'block';
            updateStatus(`Generated answer from ${data.retrieval_count} segment(s)`, 'success');
            showToast(`Answer generated using ${data.retrieval_count || 0} relevant segments`, 'success', 'Query Complete');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        let errorMessage = error.message;
        // If it's a JSON parse error, provide more helpful message
        if (errorMessage.includes('Unexpected token') || errorMessage.includes('JSON')) {
            errorMessage = 'Server returned invalid response. Please ensure the server is running and the route is registered. Error: ' + errorMessage;
        }
        updateStatus('Error: ' + errorMessage, 'error');
        showToast(errorMessage, 'error', 'Query Error');
        console.error('RAG Query error details:', error);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Ask Question';
        spinner.style.display = 'none';
    }
}

// Format RAG answer with line breaks
function formatRAGAnswer(answer) {
    // Convert newlines to <br> and preserve formatting
    return answer
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

// Display RAG sources with preview and selection
function displayRAGSources(sources, container) {
    container.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        container.innerHTML = '<p>No source segments available.</p>';
        return;
    }
    
    // Clear previous selections
    if (selectedResults.rag) {
        selectedResults.rag.clear();
    }
    
    sources.forEach((source, index) => {
        const resultId = `rag_${source.video_id}_${source.start_time}_${source.end_time}`;
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'result-item';
        sourceDiv.setAttribute('data-result-id', resultId);
        sourceDiv.innerHTML = `
            <div class="result-header">
                <label style="display: flex; align-items: center; cursor: pointer; margin-right: 10px;">
                    <input type="checkbox" class="result-checkbox" data-result-id="${resultId}" 
                           onchange="toggleSelection('${resultId}', 'rag')" style="margin-right: 8px;">
                    <span class="result-rank">#${source.rank}</span>
                </label>
                <span class="result-similarity">Relevance: ${source.similarity}</span>
            </div>
            <div class="result-content">
                <div class="result-meta">
                    <strong>Video:</strong> ${source.video_id} | 
                    <strong>Time:</strong> ${source.start_time}s - ${source.end_time}s
                </div>
                <div class="result-text">${source.text || '(No transcript text available)'}</div>
                <div class="result-actions">
                    <button class="btn-preview" onclick="previewRAGSegment('${source.video_id}', ${source.start_time}, ${source.end_time})">
                        Preview Segment
                    </button>
                    <button class="btn-view-original" onclick="openOriginalVideo('${source.video_id.replace(/'/g, "\\'")}')">
                        <i class="fas fa-external-link-alt"></i>
                        View Original Video
                    </button>
                </div>
            </div>
        `;
        container.appendChild(sourceDiv);
    });
    
    // Update selection count
    updateSelectionCountForType('rag');
}

// Preview RAG segment
async function previewRAGSegment(videoId, startTime, endTime) {
    try {
        updateStatus('Generating preview...', 'info');
        
        const response = await fetch('/api/preview-segment', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                video_id: videoId,
                start_time: startTime,
                end_time: endTime
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Show preview in modal
            showPreviewModal(data.preview_url, videoId, startTime, endTime);
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Preview error: ' + error.message);
    }
}

// Show preview modal
function showPreviewModal(previewUrl, videoId, startTime, endTime) {
    let modal = document.getElementById('previewModal');
    if (!modal) {
        modal = createPreviewModal();
    }
    
    const previewVideo = document.getElementById('previewVideo');
    const previewStatus = document.getElementById('previewStatus');
    
    previewVideo.innerHTML = `
        <video controls style="width: 100%; border-radius: 8px;">
            <source src="${previewUrl}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    `;
    previewStatus.textContent = `Preview: ${videoId} (${formatTime(startTime)} - ${formatTime(endTime)})`;
    
    modal.style.display = 'block';
}

// Show highlight player
function showHighlightPlayer(highlightUrl) {
    const statusDiv = document.getElementById('highlightStatus');
    const playerDiv = document.getElementById('highlightPlayer');
    const highlightSection = document.getElementById('highlightSection');
    
    highlightSection.style.display = 'block';
    playerDiv.innerHTML = `
        <video controls style="width: 100%; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
            <source src="${highlightUrl}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    `;
    statusDiv.textContent = '✓ Highlight reel generated!';
    statusDiv.style.background = '#d4edda';
    statusDiv.style.color = '#155724';
    
    // Scroll to highlight
    highlightSection.scrollIntoView({behavior: 'smooth'});
}

// Generate highlight from RAG selected segments
async function generateRAGHighlight() {
    const selected = Array.from(selectedResults.rag || []);
    
    if (selected.length === 0) {
        alert('Please select at least one segment to generate a highlight reel.');
        return;
    }
    
    // Convert selected IDs to result format
    const selectedSegments = [];
    for (const resultId of selected) {
        // Parse resultId: rag_videoId_startTime_endTime
        const parts = resultId.replace('rag_', '').split('_');
        if (parts.length >= 3) {
            const videoId = parts[0];
            const startTime = parseFloat(parts[parts.length - 2]);
            const endTime = parseFloat(parts[parts.length - 1]);
            
            // Find matching source from currentRAGResults
            const source = currentRAGResults.find(s => 
                s.video_id === videoId && 
                Math.abs(s.start_time - startTime) < 0.1
            );
            
            if (source) {
                selectedSegments.push({
                    video_id: source.video_id,
                    start_time: source.start_time,
                    end_time: source.end_time
                });
            }
        }
    }
    
    if (selectedSegments.length === 0) {
        alert('Could not find selected segments. Please try again.');
        return;
    }
    
    try {
        updateStatus('Generating highlight reel...', 'info');
        
        const response = await fetch('/api/generate-highlight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                results: selectedSegments,
                max_duration: 60
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showHighlightPlayer(data.highlight_url);
            updateStatus('Highlight reel generated successfully!', 'success');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Highlight generation error: ' + error.message);
    }
}

// Handle clip search
async function handleClipSearch(e) {
    e.preventDefault();
    
    // Clear previous selections for this search type
    if (typeof clearSelectionsForType === 'function') {
        clearSelectionsForType('clip');
    }
    
    const form = e.target;
    const formData = new FormData(form);
    const btn = document.getElementById('clipSearchBtn');
    const btnText = document.getElementById('clipSearchBtnText');
    const spinner = document.getElementById('clipSearchSpinner');
    const resultsContainer = document.getElementById('clipSearchResults');
    const resultsList = document.getElementById('clipResultsList');
    
    btn.disabled = true;
    btnText.textContent = 'Searching...';
    spinner.style.display = 'inline-block';
    resultsContainer.style.display = 'none';
    updateStatus('Processing clip and searching...', 'info');
    
    try {
        const response = await fetch('/api/search-clip', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentClipResults = data.results || [];
            
            // Always show results container (even if empty, to show "No Data Found" message)
            resultsContainer.style.display = 'block';
            
            // Display answer if available
            const answerBox = document.getElementById('clipAnswer');
            if (data.answer && answerBox) {
                answerBox.innerHTML = `<div class="rag-answer-content">${formatRAGAnswer(data.answer)}</div>`;
                answerBox.style.display = 'block';
            } else if (answerBox) {
                answerBox.style.display = 'none';
            }
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentClipResults, resultsList, 'clip');
            
            // Update results count
            const resultsCountEl = document.getElementById('clipResultsCount');
            if (resultsCountEl) {
                resultsCountEl.textContent = `(${data.count || 0} found)`;
            }
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateClipHighlightBtn');
            if (data.count > 0 && currentClipResults.length > 0) {
                highlightBtn.style.display = 'block';
                updateStatus(`Found ${data.count} similar segments`, 'success');
                showToast(`Found ${data.count} similar segments`, 'success', 'Search Complete');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No similar segments found', 'info');
                showToast('No similar segments found. Try a different clip.', 'info', 'No Results');
            }
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Search error: ' + error.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Search with Clip';
        spinner.style.display = 'none';
    }
}

// Display search results
function displayResults(results, container, resultsType = 'text') {
    container.innerHTML = '';
    
    if (!results || results.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'no-results';
        noResults.style.cssText = 'text-align: center; padding: 40px; color: #666;';
        noResults.innerHTML = `
            <div style="font-size: 3em; margin-bottom: 20px;">🔍</div>
            <h3 style="color: #333; margin-bottom: 10px;">No Data Found</h3>
            <p>No relevant segments found matching your query.</p>
            <p style="font-size: 0.9em; color: #888; margin-top: 10px;">Try:</p>
            <ul style="list-style: none; padding: 0; margin-top: 5px;">
                <li>• Using different keywords</li>
                <li>• Checking if videos are indexed</li>
                <li>• Adjusting your search query</li>
            </ul>
        `;
        container.appendChild(noResults);
        return;
    }
    
    // Add selection controls
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'selection-controls';
    controlsDiv.innerHTML = `
        <div style="display: flex; gap: 10px; align-items: center;">
            <button class="btn-select-all">
                <i class="fas fa-check-square"></i>
                Select All
            </button>
            <button class="btn-deselect-all">
                <i class="fas fa-square"></i>
                Clear Selection
            </button>
            <span class="selection-count">0 selected</span>
        </div>
    `;
    container.appendChild(controlsDiv);
    
    results.forEach((result, index) => {
        const resultId = `${result.video_id}_${result.start_time}_${result.end_time}`;
        const isSelected = selectedResults[resultsType] && selectedResults[resultsType].has(resultId);
        
        const item = document.createElement('div');
        item.className = 'result-item' + (isSelected ? ' selected' : '');
        item.dataset.resultId = resultId;
        item.style.cursor = 'pointer';
        item.innerHTML = `
            <div class="result-header">
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <input type="checkbox" class="result-checkbox" data-result-id="${resultId}" data-results-type="${resultsType}" ${isSelected ? 'checked' : ''} 
                           onclick="event.stopPropagation(); window.toggleSelection('${resultId.replace(/'/g, "\\'")}', '${resultsType}')">
                    <span class="result-rank">
                        <i class="fas fa-hashtag"></i>
                        ${result.rank}
                    </span>
                    <span class="similarity-badge">${(result.similarity * 100).toFixed(1)}% match</span>
                </div>
            </div>
            <div class="result-meta">
                <span>
                    <i class="fas fa-video"></i>
                    <strong>Video:</strong> ${result.video_id}
                </span>
                <span>
                    <i class="fas fa-clock"></i>
                    <strong>Time:</strong> ${formatTime(result.start_time)} - ${formatTime(result.end_time)}
                </span>
                <span>
                    <i class="fas fa-hourglass-half"></i>
                    <strong>Duration:</strong> ${result.duration}s
                </span>
            </div>
            <div class="result-text">${result.text || 'No transcript available'}</div>
            <div class="result-actions">
                <button class="btn-preview" data-video-id="${result.video_id}" data-start="${result.start_time}" data-end="${result.end_time}"
                        onclick="event.stopPropagation(); window.previewSegment('${result.video_id.replace(/'/g, "\\'")}', ${result.start_time}, ${result.end_time})">
                    <i class="fas fa-play"></i>
                    Preview Segment
                </button>
                <button class="btn-view-original" data-video-id="${result.video_id}"
                        onclick="event.stopPropagation(); window.openOriginalVideo('${result.video_id.replace(/'/g, "\\'")}')">
                    <i class="fas fa-external-link-alt"></i>
                    View Original Video
                </button>
            </div>
        `;
        
        // Click on item to toggle selection
        item.addEventListener('click', (e) => {
            if (e.target.type !== 'checkbox' && !e.target.classList.contains('btn-preview')) {
                toggleSelection(resultId, resultsType);
            }
        });
        
        container.appendChild(item);
    });
    
    // Add event listeners for selection controls
    const selectAllBtn = controlsDiv.querySelector('.btn-select-all');
    const deselectAllBtn = controlsDiv.querySelector('.btn-deselect-all');
    
    selectAllBtn.addEventListener('click', () => {
        if (!selectedResults[resultsType]) {
            selectedResults[resultsType] = new Set();
        }
        results.forEach((result) => {
            const resultId = `${result.video_id}_${result.start_time}_${result.end_time}`;
            selectedResults[resultsType].add(resultId);
        });
        updateSelectionUI(container, results, resultsType);
    });
    
    const clearBtn = controlsDiv.querySelector('.btn-deselect-all');
    clearBtn.addEventListener('click', () => {
        if (selectedResults[resultsType]) {
            results.forEach((result) => {
                const resultId = `${result.video_id}_${result.start_time}_${result.end_time}`;
                selectedResults[resultsType].delete(resultId);
            });
        }
        updateSelectionUI(container, results, resultsType);
    });
    
    updateSelectionUI(container, results, resultsType);
    
    // Initialize selection count
    updateSelectionCountForType(resultsType);
}

// Toggle selection for a result (expose globally for onclick handlers)
window.toggleSelection = function(resultId, resultsType) {
    if (!selectedResults[resultsType]) {
        selectedResults[resultsType] = new Set();
    }
    
    if (selectedResults[resultsType].has(resultId)) {
        selectedResults[resultsType].delete(resultId);
    } else {
        selectedResults[resultsType].add(resultId);
    }
    
    // Update checkbox
    const checkbox = document.querySelector(`input[data-result-id="${resultId}"]`);
    if (checkbox) {
        checkbox.checked = selectedResults[resultsType].has(resultId);
    }
    
    // Update result item styling
    const item = document.querySelector(`[data-result-id="${resultId}"]`);
    if (item) {
        if (selectedResults[resultsType].has(resultId)) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    }
    
    // Update selection count for this tab only
    updateSelectionCountForType(resultsType);
};

// Alias for local use
function toggleSelection(resultId, resultsType) {
    window.toggleSelection(resultId, resultsType);
}

// Update selection UI
function updateSelectionUI(container, results, resultsType) {
    if (!selectedResults[resultsType]) {
        selectedResults[resultsType] = new Set();
    }
    
    results.forEach((result) => {
        const resultId = `${result.video_id}_${result.start_time}_${result.end_time}`;
        const checkbox = container.querySelector(`input[data-result-id="${resultId}"]`);
        const item = container.querySelector(`[data-result-id="${resultId}"]`);
        const isSelected = selectedResults[resultsType].has(resultId);
        
        if (checkbox) {
            checkbox.checked = isSelected;
        }
        if (item) {
            if (isSelected) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        }
    });
    updateSelectionCountForType(resultsType);
}

// Update selection count display for a specific tab type
function updateSelectionCountForType(resultsType) {
    const count = selectedResults[resultsType] ? selectedResults[resultsType].size : 0;
    
    // Find the results container for this tab type
    const containerMap = {
        'clip': 'clipResultsList',
        'audio': 'audioResultsList',
        'image': 'imageResultsList',
        'rag': 'ragSourcesList'
    };
    
    const containerId = containerMap[resultsType];
    if (!containerId) return;
    
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Find selection count element directly in the container
    const countElement = container.querySelector('.selection-count');
    if (countElement) {
        countElement.textContent = `${count} selected`;
    }
    
    // Update highlight button text for this tab
    const buttonMap = {
        'clip': 'generateClipHighlightBtn',
        'audio': 'generateAudioHighlightBtn',
        'image': 'generateImageHighlightBtn',
        'rag': 'generateRAGHighlightBtn'
    };
    
    const btnId = buttonMap[resultsType];
    if (btnId) {
        const btn = document.getElementById(btnId);
        if (btn && btn.style.display !== 'none') {
            if (count > 0) {
                btn.textContent = `Generate Highlight Reel (${count} selected)`;
            } else {
                btn.textContent = 'Generate Highlight Reel (Select segments above)';
            }
        }
    }
}

// Update selection count display (global - for backward compatibility)
function updateSelectionCount() {
    // Update all tabs' selection counts
    ['text', 'clip', 'audio', 'image'].forEach(type => {
        updateSelectionCountForType(type);
    });
}

// Preview video segment (expose globally for onclick handlers)
// Open original video file in new tab
window.openOriginalVideo = function(videoId) {
    try {
        // Construct the video URL - the backend will handle finding the file
        const videoUrl = `/api/video/${encodeURIComponent(videoId)}`;
        // Open in new tab
        window.open(videoUrl, '_blank');
    } catch (error) {
        console.error('Error opening original video:', error);
        alert('Error opening video: ' + error.message);
    }
};

window.previewSegment = async function(videoId, startTime, endTime) {
    const previewModal = document.getElementById('previewModal') || createPreviewModal();
    previewModal.style.display = 'block';
    
    const previewVideo = document.getElementById('previewVideo');
    const previewStatus = document.getElementById('previewStatus');
    
    previewStatus.textContent = 'Loading preview...';
    previewVideo.innerHTML = '';
    
    try {
        const response = await fetch('/api/preview-segment', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                video_id: videoId,
                start_time: startTime,
                end_time: endTime
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            previewVideo.innerHTML = `
                <video controls style="width: 100%; border-radius: 8px;">
                    <source src="${data.preview_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            previewStatus.textContent = `Preview: ${videoId} (${formatTime(startTime)} - ${formatTime(endTime)})`;
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        previewStatus.textContent = 'Error: ' + error.message;
        previewVideo.innerHTML = '<p style="color: red;">Failed to load preview</p>';
    }
};

// Alias for local use
async function previewSegment(videoId, startTime, endTime) {
    return window.previewSegment(videoId, startTime, endTime);
}

// Create preview modal
function createPreviewModal() {
    const modal = document.createElement('div');
    modal.id = 'previewModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8);';
    
    modal.innerHTML = `
        <div style="position: relative; background: white; margin: 5% auto; padding: 20px; border-radius: 10px; width: 80%; max-width: 800px;">
            <span class="close-preview" style="position: absolute; right: 20px; top: 15px; font-size: 28px; font-weight: bold; cursor: pointer; color: #aaa;">&times;</span>
            <h2 style="margin-bottom: 15px;">Video Preview</h2>
            <div id="previewStatus" style="margin-bottom: 10px; color: #666;"></div>
            <div id="previewVideo"></div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal
    modal.querySelector('.close-preview').addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    return modal;
}

// Format time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Play full video (expose globally for onclick handlers)
window.playVideo = function(videoUrl, videoTitle) {
    let modal = document.getElementById('videoPlayerModal');
    if (!modal) {
        modal = createVideoPlayerModal();
    }
    
    const playerVideo = document.getElementById('playerVideo');
    const playerTitle = document.getElementById('playerTitle');
    const playerStatus = document.getElementById('playerStatus');
    
    if (playerTitle) {
        playerTitle.textContent = videoTitle || 'Video Player';
    }
    
    if (playerStatus) {
        playerStatus.textContent = 'Loading video...';
    }
    
    if (playerVideo) {
        playerVideo.innerHTML = '';
        const video = document.createElement('video');
        video.controls = true;
        video.style.width = '100%';
        video.style.borderRadius = '8px';
        video.style.maxHeight = '80vh';
        video.onloadstart = () => {
            if (playerStatus) playerStatus.textContent = 'Loading video...';
        };
        video.oncanplay = () => {
            if (playerStatus) playerStatus.textContent = 'Ready to play';
        };
        video.onerror = () => {
            if (playerStatus) playerStatus.textContent = 'Error loading video';
            playerVideo.innerHTML = '<p style="color: red; padding: 20px;">Failed to load video. Please check if the file exists.</p>';
        };
        
        const source = document.createElement('source');
        source.src = videoUrl;
        source.type = 'video/mp4';
        video.appendChild(source);
        playerVideo.appendChild(video);
    }
    
    modal.style.display = 'block';
};

// Alias for local use
function playVideo(videoUrl, videoTitle) {
    window.playVideo(videoUrl, videoTitle);
}

// Create video player modal
function createVideoPlayerModal() {
    const modal = document.createElement('div');
    modal.id = 'videoPlayerModal';
    modal.style.cssText = 'display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); overflow: auto;';
    
    modal.innerHTML = `
        <div style="position: relative; background: #1a1a1a; margin: 2% auto; padding: 20px; border-radius: 10px; width: 90%; max-width: 1200px; color: white;">
            <span class="close-video-player" style="position: absolute; right: 20px; top: 15px; font-size: 32px; font-weight: bold; cursor: pointer; color: #aaa; z-index: 1001;">&times;</span>
            <h2 id="playerTitle" style="margin-bottom: 15px; color: white;">Video Player</h2>
            <div id="playerStatus" style="margin-bottom: 10px; color: #aaa; font-size: 14px;"></div>
            <div id="playerVideo" style="background: #000; border-radius: 8px; padding: 10px; min-height: 400px; display: flex; align-items: center; justify-content: center;"></div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal
    const closeBtn = modal.querySelector('.close-video-player');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
            // Stop video playback
            const video = modal.querySelector('video');
            if (video) {
                video.pause();
                video.src = '';
            }
        });
    }
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
            // Stop video playback
            const video = modal.querySelector('video');
            if (video) {
                video.pause();
                video.src = '';
            }
        }
    });
    
    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = 'none';
            const video = modal.querySelector('video');
            if (video) {
                video.pause();
                video.src = '';
            }
        }
    });
    
    return modal;
}

// Get selected results from current search results for a specific tab
function getSelectedResults(resultsArray, resultsType) {
    const selected = [];
    
    if (!resultsType || !selectedResults[resultsType]) {
        return selected;
    }
    
    resultsArray.forEach(result => {
        const resultId = `${result.video_id}_${result.start_time}_${result.end_time}`;
        if (selectedResults[resultsType].has(resultId)) {
            selected.push(result);
        }
    });
    return selected;
}

// Generate highlight reel
async function generateHighlight() {
    // Get selected results for text search
    const selected = getSelectedResults(currentResults || [], 'text');
    
    if (selected.length === 0) {
        alert('Please select at least one result to generate highlight from. Use checkboxes to select segments.');
        return;
    }
    
    const maxDuration = prompt(`Maximum duration in seconds? (${selected.length} segment(s) selected)`, '60');
    if (!maxDuration) return;
    
    const statusDiv = document.getElementById('highlightStatus');
    const playerDiv = document.getElementById('highlightPlayer');
    const highlightSection = document.getElementById('highlightSection');
    
    highlightSection.style.display = 'block';
    statusDiv.textContent = `Generating highlight reel from ${selected.length} selected segment(s)... This may take a minute.`;
    playerDiv.innerHTML = '';
    
    try {
        const response = await fetch('/api/generate-highlight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                results: selected,
                max_duration: parseInt(maxDuration)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            playerDiv.innerHTML = `
                <video controls>
                    <source src="${data.highlight_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            statusDiv.textContent = '✓ Highlight reel generated!';
            statusDiv.style.background = '#d4edda';
            statusDiv.style.color = '#155724';
            
            // Scroll to highlight
            highlightSection.scrollIntoView({behavior: 'smooth'});
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
        statusDiv.style.background = '#f8d7da';
        statusDiv.style.color = '#721c24';
    }
}

// Generate highlight from clip search
async function generateClipHighlight() {
    const selected = getSelectedResults(currentClipResults || [], 'clip');
    if (selected.length === 0) {
        alert('Please select at least one result to generate highlight from. Use checkboxes to select segments.');
        return;
    }
    
    // Use the general highlight generation but with selected clip results
    const maxDuration = prompt(`Maximum duration in seconds? (${selected.length} segment(s) selected)`, '60');
    if (!maxDuration) return;
    
    const statusDiv = document.getElementById('highlightStatus');
    const playerDiv = document.getElementById('highlightPlayer');
    const highlightSection = document.getElementById('highlightSection');
    
    highlightSection.style.display = 'block';
    statusDiv.textContent = `Generating highlight reel from ${selected.length} selected segment(s)... This may take a minute.`;
    playerDiv.innerHTML = '';
    
    try {
        const response = await fetch('/api/generate-highlight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                results: selected,
                max_duration: parseInt(maxDuration)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            playerDiv.innerHTML = `
                <video controls>
                    <source src="${data.highlight_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            statusDiv.textContent = '✓ Highlight reel generated!';
            statusDiv.style.background = '#d4edda';
            statusDiv.style.color = '#155724';
            highlightSection.scrollIntoView({behavior: 'smooth'});
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
        statusDiv.style.background = '#f8d7da';
        statusDiv.style.color = '#721c24';
    }
}

// Handle audio search
async function handleAudioSearch(e) {
    e.preventDefault();
    
    // Clear previous selections for this search type
    if (typeof clearSelectionsForType === 'function') {
        clearSelectionsForType('audio');
    }
    
    const form = e.target;
    const formData = new FormData(form);
    const btn = document.getElementById('audioSearchBtn');
    const btnText = document.getElementById('audioSearchBtnText');
    const spinner = document.getElementById('audioSearchSpinner');
    const resultsContainer = document.getElementById('audioSearchResults');
    const resultsList = document.getElementById('audioResultsList');
    
    btn.disabled = true;
    btnText.textContent = 'Searching...';
    spinner.style.display = 'inline-block';
    resultsContainer.style.display = 'none';
    updateStatus('Processing audio and searching...', 'info');
    
    try {
        const response = await fetch('/api/search-audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentAudioResults = data.results || [];
            
            // Always show results container (even if empty, to show "No Data Found" message)
            resultsContainer.style.display = 'block';
            
            // Display answer if available
            const answerBox = document.getElementById('audioAnswer');
            if (data.answer && answerBox) {
                answerBox.innerHTML = `<div class="rag-answer-content">${formatRAGAnswer(data.answer)}</div>`;
                answerBox.style.display = 'block';
            } else if (answerBox) {
                answerBox.style.display = 'none';
            }
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentAudioResults, resultsList, 'audio');
            
            // Update results count
            const resultsCountEl = document.getElementById('audioResultsCount');
            if (resultsCountEl) {
                resultsCountEl.textContent = `(${data.count || 0} found)`;
            }
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateAudioHighlightBtn');
            if (data.count > 0 && currentAudioResults.length > 0) {
                highlightBtn.style.display = 'block';
                updateStatus(`Found ${data.count} relevant segments`, 'success');
                showToast(`Found ${data.count} relevant segments`, 'success', 'Search Complete');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No relevant segments found', 'info');
                showToast('No relevant segments found. Try a different audio file.', 'info', 'No Results');
            }
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Search error: ' + error.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Search with Audio';
        spinner.style.display = 'none';
    }
}

// Handle image search
async function handleImageSearch(e) {
    e.preventDefault();
    
    // Clear previous selections for this search type
    if (typeof clearSelectionsForType === 'function') {
        clearSelectionsForType('image');
    }
    
    const form = e.target;
    const formData = new FormData(form);
    const btn = document.getElementById('imageSearchBtn');
    const btnText = document.getElementById('imageSearchBtnText');
    const spinner = document.getElementById('imageSearchSpinner');
    const resultsContainer = document.getElementById('imageSearchResults');
    const resultsList = document.getElementById('imageResultsList');
    
    btn.disabled = true;
    btnText.textContent = 'Searching...';
    spinner.style.display = 'inline-block';
    resultsContainer.style.display = 'none';
    updateStatus('Processing image and searching...', 'info');
    
    try {
        const response = await fetch('/api/search-image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentImageResults = data.results || [];
            
            // Always show results container (even if empty, to show "No Data Found" message)
            resultsContainer.style.display = 'block';
            
            // Display answer if available
            const answerBox = document.getElementById('imageAnswer');
            if (data.answer && answerBox) {
                answerBox.innerHTML = `<div class="rag-answer-content">${formatRAGAnswer(data.answer)}</div>`;
                answerBox.style.display = 'block';
            } else if (answerBox) {
                answerBox.style.display = 'none';
            }
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentImageResults, resultsList, 'image');
            
            // Update results count
            const resultsCountEl = document.getElementById('imageResultsCount');
            if (resultsCountEl) {
                resultsCountEl.textContent = `(${data.count || 0} found)`;
            }
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateImageHighlightBtn');
            if (data.count > 0 && currentImageResults.length > 0) {
                highlightBtn.style.display = 'block';
                updateStatus(`Found ${data.count} visually similar segments`, 'success');
                showToast(`Found ${data.count} visually similar segments`, 'success', 'Search Complete');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No visually similar segments found', 'info');
                showToast('No visually similar segments found. Try a different image.', 'info', 'No Results');
            }
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
        alert('Search error: ' + error.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Search with Image';
        spinner.style.display = 'none';
    }
}

// Generate highlight from audio search
async function generateAudioHighlight() {
    const selected = getSelectedResults(currentAudioResults || [], 'audio');
    if (selected.length === 0) {
        alert('Please select at least one result to generate highlight from. Use checkboxes to select segments.');
        return;
    }
    
    const maxDuration = prompt(`Maximum duration in seconds? (${selected.length} segment(s) selected)`, '60');
    if (!maxDuration) return;
    
    const statusDiv = document.getElementById('highlightStatus');
    const playerDiv = document.getElementById('highlightPlayer');
    const highlightSection = document.getElementById('highlightSection');
    
    highlightSection.style.display = 'block';
    statusDiv.textContent = `Generating highlight reel from ${selected.length} selected segment(s)... This may take a minute.`;
    playerDiv.innerHTML = '';
    
    try {
        const response = await fetch('/api/generate-highlight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                results: selected,
                max_duration: parseInt(maxDuration)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            playerDiv.innerHTML = `
                <video controls>
                    <source src="${data.highlight_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            statusDiv.textContent = '✓ Highlight reel generated!';
            statusDiv.style.background = '#d4edda';
            statusDiv.style.color = '#155724';
            highlightSection.scrollIntoView({behavior: 'smooth'});
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
        statusDiv.style.background = '#f8d7da';
        statusDiv.style.color = '#721c24';
    }
}

// Generate highlight from image search
async function generateImageHighlight() {
    const selected = getSelectedResults(currentImageResults || [], 'image');
    if (selected.length === 0) {
        alert('Please select at least one result to generate highlight from. Use checkboxes to select segments.');
        return;
    }
    
    const maxDuration = prompt(`Maximum duration in seconds? (${selected.length} segment(s) selected)`, '60');
    if (!maxDuration) return;
    
    const statusDiv = document.getElementById('highlightStatus');
    const playerDiv = document.getElementById('highlightPlayer');
    const highlightSection = document.getElementById('highlightSection');
    
    highlightSection.style.display = 'block';
    statusDiv.textContent = `Generating highlight reel from ${selected.length} selected segment(s)... This may take a minute.`;
    playerDiv.innerHTML = '';
    
    try {
        const response = await fetch('/api/generate-highlight', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                results: selected,
                max_duration: parseInt(maxDuration)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            playerDiv.innerHTML = `
                <video controls>
                    <source src="${data.highlight_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            `;
            statusDiv.textContent = '✓ Highlight reel generated!';
            statusDiv.style.background = '#d4edda';
            statusDiv.style.color = '#155724';
            highlightSection.scrollIntoView({behavior: 'smooth'});
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
        statusDiv.style.background = '#f8d7da';
        statusDiv.style.color = '#721c24';
    }
}

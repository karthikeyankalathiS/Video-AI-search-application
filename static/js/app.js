// Global state
let currentResults = [];
let currentClipResults = [];
let currentAudioResults = [];
let currentImageResults = [];

// Separate selection tracking for each tab
let selectedResults = {
    text: new Set(),
    clip: new Set(),
    audio: new Set(),
    image: new Set()
};

// Clear selections for a specific type (define early so it's available to all handlers)
function clearSelectionsForType(type) {
    if (selectedResults[type]) {
        selectedResults[type].clear();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setupEventListeners();
});

// Check agent status
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.agent_initialized) {
            updateStatus('Agent ready', 'success');
            if (data.index_loaded) {
                updateStatus(`Index loaded: ${data.videos_count} video(s), ${data.segments_count} segments`, 'info');
            } else {
                updateStatus('Agent ready. Please index videos to start searching.', 'info');
            }
        } else {
            updateStatus('Initializing agent...', 'info');
            // Initialize agent
            await fetch('/api/initialize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_size: 'base'})
            });
            updateStatus('Agent initialized. Please index videos.', 'success');
        }
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('indexForm').addEventListener('submit', handleIndex);
    document.getElementById('searchForm').addEventListener('submit', handleSearch);
    document.getElementById('clipSearchForm').addEventListener('submit', handleClipSearch);
    document.getElementById('audioSearchForm').addEventListener('submit', handleAudioSearch);
    document.getElementById('imageSearchForm').addEventListener('submit', handleImageSearch);
    document.getElementById('generateHighlightBtn').addEventListener('click', generateHighlight);
    document.getElementById('generateClipHighlightBtn').addEventListener('click', generateClipHighlight);
    document.getElementById('generateAudioHighlightBtn').addEventListener('click', generateAudioHighlight);
    document.getElementById('generateImageHighlightBtn').addEventListener('click', generateImageHighlight);
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
        'text-search': 'text',
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
    statusBar.className = 'status-bar ' + type;
    statusText.textContent = message;
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
    
    btn.disabled = true;
    btnText.textContent = 'Indexing...';
    spinner.style.display = 'inline-block';
    resultBox.style.display = 'none';
    updateStatus('Indexing videos... This may take a few minutes.', 'info');
    
    try {
        const response = await fetch('/api/index', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            resultBox.style.display = 'block';
            resultBox.innerHTML = `
                <h4>✓ Indexing Complete!</h4>
                <p><strong>Videos:</strong> ${data.videos_count}</p>
                <p><strong>Segments:</strong> ${data.segments_count}</p>
                <p><strong>Index saved:</strong> ${data.index_path}</p>
            `;
            resultBox.style.borderLeftColor = '#28a745';
            updateStatus(`Indexed ${data.videos_count} video(s) with ${data.segments_count} segments`, 'success');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        resultBox.style.display = 'block';
        resultBox.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        resultBox.style.borderLeftColor = '#dc3545';
        updateStatus('Error: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Index Videos';
        spinner.style.display = 'none';
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
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentClipResults, resultsList, 'clip');
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateClipHighlightBtn');
            if (data.count > 0 && currentClipResults.length > 0) {
                highlightBtn.style.display = 'block';
                updateStatus(`Found ${data.count} similar segments`, 'success');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No similar segments found', 'info');
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
    controlsDiv.style.cssText = 'margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;';
    controlsDiv.innerHTML = `
        <div>
            <button class="btn-select-all" style="margin-right: 10px;">Select All</button>
            <button class="btn-deselect-all" style="margin-right: 10px;">Clear Selection</button>
            <span class="selection-count" style="color: #666; font-weight: 600;">0 selected</span>
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
                <div style="display: flex; align-items: center; gap: 10px;">
                    <input type="checkbox" class="result-checkbox" data-result-id="${resultId}" data-results-type="${resultsType}" ${isSelected ? 'checked' : ''} 
                           onclick="event.stopPropagation(); window.toggleSelection('${resultId.replace(/'/g, "\\'")}', '${resultsType}')">
                    <span class="result-rank">#${result.rank}</span>
                    <span class="similarity-badge">Similarity: ${result.similarity}</span>
                </div>
            </div>
            <div class="result-meta">
                <span><strong>Video:</strong> ${result.video_id}</span>
                <span><strong>Time:</strong> ${formatTime(result.start_time)} - ${formatTime(result.end_time)}</span>
                <span><strong>Duration:</strong> ${result.duration}s</span>
            </div>
            <div class="result-text">${result.text}</div>
            <div class="result-actions" style="margin-top: 10px;">
                <button class="btn-preview" data-video-id="${result.video_id}" data-start="${result.start_time}" data-end="${result.end_time}"
                        onclick="event.stopPropagation(); window.previewSegment('${result.video_id.replace(/'/g, "\\'")}', ${result.start_time}, ${result.end_time})">
                    ▶ Preview Segment
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
        'text': 'resultsList',
        'clip': 'clipResultsList',
        'audio': 'audioResultsList',
        'image': 'imageResultsList'
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
        'text': 'generateHighlightBtn',
        'clip': 'generateClipHighlightBtn',
        'audio': 'generateAudioHighlightBtn',
        'image': 'generateImageHighlightBtn'
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
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentAudioResults, resultsList, 'audio');
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateAudioHighlightBtn');
            if (data.count > 0 && currentAudioResults.length > 0) {
                highlightBtn.style.display = 'block';
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
            
            // Display results (will show "No Data Found" if empty)
            displayResults(currentImageResults, resultsList, 'image');
            
            // Show/hide highlight button based on results
            const highlightBtn = document.getElementById('generateImageHighlightBtn');
            if (data.count > 0 && currentImageResults.length > 0) {
                highlightBtn.style.display = 'block';
                updateStatus(`Found ${data.count} visually similar segments`, 'success');
            } else {
                highlightBtn.style.display = 'none';
                updateStatus('No visually similar segments found', 'info');
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

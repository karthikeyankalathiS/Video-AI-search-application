#!/bin/bash
# Start Flask server with OpenAI support

cd /Users/karthi/Documents/Hackathon/videoAi

# Activate virtual environment
source venv/bin/activate

# Kill any existing server on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null
sleep 2

# Start the server
echo "Starting Flask server with OpenAI embeddings..."
python3 app.py > server.log 2>&1 &
SERVER_PID=$!

# Save PID
echo $SERVER_PID > server.pid

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Server started successfully (PID: $SERVER_PID)"
    echo "✓ Server running at: http://localhost:5001"
    echo ""
    echo "To stop the server: kill $SERVER_PID"
    echo "To view logs: tail -f server.log"
else
    echo "✗ Server failed to start. Check server.log for errors."
    exit 1
fi


# RAG (Retrieval-Augmented Generation) Implementation

## Overview
This document describes the RAG implementation added to the Video Search Agent system. RAG combines retrieval (finding relevant video segments) with generation (using LLM to generate comprehensive answers).

## Architecture

### Components

1. **RAG Agent (`rag_agent.py`)**
   - Uses the existing `VideoSearchAgent` for retrieval
   - Integrates with OpenAI API for answer generation
   - Falls back to template-based responses if OpenAI is unavailable
   - Supports custom context segments

2. **API Endpoint (`/api/rag-query`)**
   - Accepts natural language questions
   - Retrieves relevant video segments
   - Generates contextual answers using LLM
   - Returns answer with source segments

3. **Frontend Integration**
   - New "RAG Query" tab in the web UI
   - Question input with model selection
   - Answer display with source citations
   - Configurable retrieval parameters

## Features

### Retrieval
- Uses existing semantic search to find relevant video segments
- Configurable number of segments to retrieve (top_k)
- Filters by similarity threshold

### Generation
- **OpenAI Integration**: Uses GPT-3.5 Turbo or GPT-4 for high-quality answers
- **Template Fallback**: Simple template-based responses when OpenAI is unavailable
- **Context-Aware**: Answers are generated based on retrieved video segments
- **Source Citations**: Each answer includes references to source segments

### Configuration Options
- **Model Selection**: Choose between GPT-3.5 Turbo, GPT-4, or GPT-4 Turbo
- **Retrieval Count**: Adjust number of segments to retrieve (1-20)
- **OpenAI Toggle**: Enable/disable OpenAI usage
- **API Key**: Optional API key input (uses environment variable by default)

## Usage

### API Usage

```python
POST /api/rag-query
Content-Type: application/json

{
    "question": "What is dependency injection?",
    "top_k": 5,
    "model": "gpt-3.5-turbo",
    "use_openai": true,
    "openai_api_key": "optional-key"
}
```

### Response Format

```json
{
    "status": "success",
    "question": "What is dependency injection?",
    "answer": "Based on the video content...",
    "sources": [
        {
            "rank": 1,
            "video_id": "video1",
            "start_time": 10.5,
            "end_time": 25.3,
            "text": "Dependency injection is...",
            "similarity": 0.85
        }
    ],
    "retrieval_count": 5
}
```

### Web UI Usage

1. Navigate to the "RAG Query" tab
2. Enter your question in the text area
3. Configure retrieval and model settings
4. Click "Ask Question"
5. View the generated answer with source segments

## Implementation Details

### RAG Agent Class

The `RAGAgent` class provides:

- `query(question, top_k, max_context_segments)`: Main query method
- `query_with_custom_context(question, custom_segments)`: Query with custom segments
- `_format_context(segments)`: Formats segments for LLM context
- `_generate_with_openai(question, context)`: OpenAI-based generation
- `_generate_template_answer(question, segments)`: Template-based fallback

### Integration Points

1. **Video Search Agent**: Uses existing search functionality for retrieval
2. **Flask App**: New endpoint integrated into existing app structure
3. **Frontend**: New tab integrated with existing UI components

## Dependencies

- `openai>=1.0.0` (already in requirements.txt)
- Existing dependencies (sentence-transformers, whisper, etc.)

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (optional, can be provided in UI)

## Error Handling

- Graceful fallback to template-based responses if OpenAI fails
- Clear error messages for missing index or initialization issues
- Validation for required fields

## Future Enhancements

Potential improvements:
- Support for other LLM providers (Anthropic, local models)
- Multi-turn conversation support
- Answer quality scoring
- Context window optimization
- Streaming responses for better UX

## Notes

- RAG works best when videos are properly indexed with transcripts
- OpenAI API usage incurs costs (configurable)
- Template-based mode provides basic functionality without API costs
- Source segments help verify answer accuracy


#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Agent for Video Search
Uses retrieved video segments as context to generate comprehensive answers
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI


class RAGAgent:
    """
    RAG Agent that combines retrieval (video search) with generation (LLM).
    Retrieves relevant video segments and uses them as context for answer generation.
    """
    
    def __init__(self, video_search_agent, openai_api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo", use_openai: bool = True):
        """
        Initialize RAG Agent.
        
        Args:
            video_search_agent: Instance of VideoSearchAgent for retrieval
            openai_api_key: OpenAI API key (or from environment)
            model: LLM model to use (gpt-3.5-turbo, gpt-4, etc.)
            use_openai: Whether to use OpenAI (True) or fallback to simple template
        """
        self.video_search_agent = video_search_agent
        self.model = model
        self.use_openai = use_openai
        
        # Initialize OpenAI client if available and requested
        self.openai_client = None
        if use_openai:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✓ RAG Agent initialized with OpenAI")
                except Exception as e:
                    print(f"⚠ Warning: Could not initialize OpenAI for RAG: {e}")
                    print("  Falling back to template-based responses")
                    self.use_openai = False
            else:
                print("⚠ Warning: OPENAI_API_KEY not set. Using template-based responses")
                self.use_openai = False
    
    def query(self, question: str, top_k: int = 5, max_context_segments: int = 10) -> Dict:
        """
        Answer a question using RAG: retrieve relevant segments and generate answer.
        
        Args:
            question: User's question
            top_k: Number of segments to retrieve
            max_context_segments: Maximum number of segments to include in context
            
        Returns:
            Dictionary with answer, retrieved segments, and metadata
        """
        print(f"\n🔍 RAG Query: '{question}'")
        
        # Step 1: Retrieve relevant segments
        print("  [1/2] Retrieving relevant video segments...")
        search_results = self.video_search_agent.search_by_text(question, top_k=top_k)
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant video segments to answer your question. Please try rephrasing your query or ensure videos have been indexed.",
                'segments': [],
                'sources': [],
                'retrieval_count': 0
            }
        
        # Step 2: Format context from retrieved segments
        context_segments = search_results[:max_context_segments]
        context_text = self._format_context(context_segments)
        
        # Step 3: Generate answer using LLM
        print("  [2/2] Generating answer with LLM...")
        if self.use_openai and self.openai_client:
            answer = self._generate_with_openai(question, context_text)
        else:
            answer = self._generate_template_answer(question, context_segments)
        
        # Format sources with all available text
        sources = []
        for i, result in enumerate(context_segments, 1):
            segment = result.get('segment', {})
            text = segment.get('text', '').strip()
            ocr_text = segment.get('on_screen_text', '').strip()
            combined_text = segment.get('combined_text', '').strip()
            
            # Use the best available text
            display_text = combined_text or text or ocr_text
            if display_text:
                display_text = display_text[:200] + '...' if len(display_text) > 200 else display_text
            else:
                display_text = '(No transcript text available)'
            
            sources.append({
                'rank': i,
                'video_id': result['video_id'],
                'start_time': round(result['start'], 2),
                'end_time': round(result['end'], 2),
                'text': display_text,
                'similarity': round(result['similarity'], 3)
            })
        
        print(f"  ✓ Generated answer from {len(context_segments)} segment(s)")
        
        return {
            'answer': answer,
            'segments': context_segments,
            'sources': sources,
            'retrieval_count': len(context_segments),
            'question': question
        }
    
    def _format_context(self, segments: List[Dict]) -> str:
        """Format retrieved segments into context text for LLM."""
        context_parts = []
        
        for i, result in enumerate(segments, 1):
            video_id = result['video_id']
            start = result['start']
            end = result['end']
            segment = result.get('segment', {})
            text = segment.get('text', '').strip()
            ocr_text = segment.get('on_screen_text', '').strip()
            combined_text = segment.get('combined_text', '').strip()
            similarity = result['similarity']
            
            # Use combined text if available, otherwise use text or OCR
            content_text = combined_text or text or ocr_text
            
            if content_text:
                context_parts.append(
                    f"[Segment {i} from video '{video_id}' (time: {start:.1f}s-{end:.1f}s, relevance: {similarity:.2f})]\n"
                    f"{content_text}\n"
                )
            else:
                # Even if no text, include segment info for context
                context_parts.append(
                    f"[Segment {i} from video '{video_id}' (time: {start:.1f}s-{end:.1f}s, relevance: {similarity:.2f})]\n"
                    f"Visual content only - no transcript available for this segment.\n"
                )
        
        return "\n".join(context_parts)
    
    def _generate_with_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            system_prompt = """You are a helpful assistant that answers questions based on video transcript segments. 
Use the provided video segments as context to answer the user's question accurately and comprehensively.
- Provide detailed, informative answers that directly address the question
- Include specific details, facts, and information from the video segments
- If the context doesn't contain enough information to answer the question, clearly state what information is available and what is missing
- Cite specific segments when relevant (e.g., "According to segment 1 at 270s...")
- Structure your answer clearly with relevant details
- Be thorough but concise"""
            
            user_prompt = f"""Based on the following video transcript segments, please provide a detailed answer to this question:

Question: {question}

Video Segments:
{context}

Please provide a comprehensive and detailed answer based on the video content above. Include specific information, details, and facts from the segments that are relevant to answering the question."""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800  # Increased for more detailed answers
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  ⚠ Error generating answer with OpenAI: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _generate_template_answer(self, question: str, segments: List[Dict]) -> str:
        """Generate a detailed template-based answer when LLM is not available."""
        if not segments:
            return "No relevant segments found to answer this question."
        
        # Extract key information from segments
        answer_parts = [f"Based on the video content, here's what I found regarding '{question}':\n"]
        
        # Collect all available text from segments
        segment_texts = []
        for i, result in enumerate(segments, 1):
            segment = result.get('segment', {})
            text = segment.get('text', '').strip()
            ocr_text = segment.get('on_screen_text', '').strip()
            combined_text = segment.get('combined_text', '').strip()
            
            # Use the best available text
            content_text = combined_text or text or ocr_text
            
            if content_text:
                video_id = result.get('video_id', 'unknown')
                start = result.get('start', 0)
                end = result.get('end', 0)
                similarity = result.get('similarity', 0)
                
                segment_texts.append({
                    'text': content_text,
                    'video_id': video_id,
                    'time': f"{start:.1f}s-{end:.1f}s",
                    'similarity': similarity
                })
        
        if segment_texts:
            # Include detailed information from segments
            answer_parts.append("Here are the relevant details from the video:\n")
            
            for i, seg_info in enumerate(segment_texts[:5], 1):  # Use top 5 segments
                answer_parts.append(
                    f"\n[{i}] From video '{seg_info['video_id']}' (time: {seg_info['time']}, relevance: {seg_info['similarity']:.2f}):\n"
                    f"{seg_info['text'][:300]}{'...' if len(seg_info['text']) > 300 else ''}"
                )
            
            # Summary
            answer_parts.append(
                f"\n\nSummary: Found {len(segments)} relevant segment(s) in the video corpus. "
                f"The above information is extracted from the most relevant segments that match your question."
            )
        else:
            # No text available - provide helpful message
            answer_parts.append(
                f"\nI found {len(segments)} relevant segment(s) in the video corpus, but these segments don't have transcript text available. "
                f"This might be a mute video or the audio wasn't transcribed. "
                f"You can use the 'Preview Segment' buttons below to watch the relevant video segments directly."
            )
        
        return "\n".join(answer_parts)
    
    def query_with_custom_context(self, question: str, custom_segments: List[Dict]) -> Dict:
        """
        Answer a question using custom segments (not retrieved via search).
        Useful for answering questions about specific video segments.
        
        Args:
            question: User's question
            custom_segments: List of segment dictionaries to use as context
            
        Returns:
            Dictionary with answer and metadata
        """
        if not custom_segments:
            return {
                'answer': "No segments provided for context.",
                'segments': [],
                'sources': [],
                'retrieval_count': 0
            }
        
        context_text = self._format_context(custom_segments)
        
        if self.use_openai and self.openai_client:
            answer = self._generate_with_openai(question, context_text)
        else:
            answer = self._generate_template_answer(question, custom_segments)
        
        sources = []
        for i, result in enumerate(custom_segments, 1):
            sources.append({
                'rank': i,
                'video_id': result.get('video_id', 'unknown'),
                'start_time': round(result.get('start', 0), 2),
                'end_time': round(result.get('end', 0), 2),
                'text': result.get('segment', {}).get('text', '')[:200] + '...' if len(result.get('segment', {}).get('text', '')) > 200 else result.get('segment', {}).get('text', ''),
                'similarity': round(result.get('similarity', 0), 3) if 'similarity' in result else None
            })
        
        return {
            'answer': answer,
            'segments': custom_segments,
            'sources': sources,
            'retrieval_count': len(custom_segments),
            'question': question
        }


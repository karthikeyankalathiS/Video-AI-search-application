#!/usr/bin/env python3
"""
Quick verification script to check OpenAI setup and test embeddings
"""
import sys
import os

print("="*70)
print("Verifying OpenAI Setup")
print("="*70)

# Check OpenAI package
try:
    from openai import OpenAI
    print("✓ OpenAI package imported successfully")
except ImportError as e:
    print(f"✗ OpenAI package not found: {e}")
    print("  Run: pip install openai")
    sys.exit(1)

# Check API key
api_key = os.getenv('OPENAI_API_KEY')

if not api_key or api_key.startswith('sk-'):
    print("✓ OpenAI API key is set")
else:
    print("⚠ OpenAI API key may not be set correctly")

# Test OpenAI client initialization
try:
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Test embedding generation
try:
    print("\nTesting embedding generation...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="test query"
    )
    embedding = response.data[0].embedding
    print(f"✓ Embedding generated successfully (dimension: {len(embedding)})")
except Exception as e:
    print(f"✗ Failed to generate embedding: {e}")
    sys.exit(1)

# Check VideoSearchAgent
try:
    from video_search_agent import VideoSearchAgent
    print("\n✓ VideoSearchAgent module imported")
    
    print("\nInitializing VideoSearchAgent with OpenAI...")
    agent = VideoSearchAgent(
        model_size='base',
        use_openai=True,
        openai_api_key=api_key
    )
    print("✓ VideoSearchAgent initialized with OpenAI embeddings")
    print(f"✓ Similarity threshold: {agent.similarity_threshold}")
    
except Exception as e:
    print(f"\n✗ Failed to initialize VideoSearchAgent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ All checks passed! OpenAI setup is ready.")
print("="*70)
print("\nNext steps:")
print("1. Restart Flask server: python3 app.py")
print("2. Re-index your videos for best results")
print("3. Test search queries")


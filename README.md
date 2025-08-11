# Basic Chat Example

A simple example showing how to use the Cartesia Voice Agents SDK with Google Gemini.

## Setup

1. Set up your environment variables:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Install dependencies and run:

   **Option A: Using uv (recommended)**
   ```bash
   uv run main.py
   ```

   **Option B: Using pip with virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install from pyproject.toml (includes SDK and all dependencies)
   pip install -e .

   # Run the example
   python main.py
   ```

## Features

- Voice conversation using Cartesia's voice agent system
- Google Gemini for natural language processing
- End call functionality through system tools

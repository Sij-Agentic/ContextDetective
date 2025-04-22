# Context Detective

Context Detective is an AI-powered application that analyzes images to determine their context, meaning, and significance. It uses a combination of visual analysis, style recognition, web search, and memory-enhanced inference to provide comprehensive insights about images.

## Features

- **Visual Elements Analysis**: Identifies objects, people, colors, and text in images
- **Style Analysis**: Recognizes artistic styles and cultural elements
- **Scenario Analysis**: Determines what might be happening in the image
- **Context Inference**: Provides a comprehensive analysis of the image's context
- **Confidence Rating**: Indicates how confident the system is in its analysis
- **Memory System**: Stores and retrieves analyses using vector-based similarity search
- **Exact Match Caching**: Instantly retrieves results for previously analyzed images

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/context-detective.git
   cd context-detective
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   MEMORY_PATH=path/to/memory_storage  # Optional: defaults to "memory_storage"
   OLLAMA_URL=http://localhost:11434   # Optional: for vector embeddings
   ```

## Usage

1. Start the MCP server:
   ```
   python main.py
   ```

2. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Upload an image using the file uploader.

4. Click the "Analyze" button to start the analysis.

5. View the results, which include:
   - Context Guess
   - Confidence Rating
   - Explanation
   - Related Links
   - Search Terms Used

## How It Works

Context Detective uses a multi-step workflow to analyze images:

1. **Hash Computation**: Generates a unique hash for the image and checks for exact matches in memory
2. **Visual Elements Analysis**: Identifies objects, people, colors, and text in the image
3. **Style Analysis**: Recognizes artistic styles and cultural elements
4. **Scenario Analysis**: Determines what might be happening in the image
5. **Search Term Generation**: Creates effective search terms based on the analyses
6. **Web Search**: Gathers contextual information from the web
7. **Memory Retrieval**: Finds semantically similar past analyses using vector search
8. **Context Inference**: Combines all analyses and memory to determine the image's context
9. **Memory Storage**: Stores the complete analysis in vector database for future use

## Memory System

Context Detective features a sophisticated memory system with multiple layers:

### Short-Term Memory

- **Session-based storage**: Keeps track of intermediate analysis results during a session
- **Component caching**: Stores visual elements, style analysis, and scenario analysis separately
- **Reduces redundant processing**: Avoids re-analyzing components within the same session

### Long-Term Memory (ChromaDB)

- **Vector storage**: Converts analyses into embeddings for semantic similarity search
- **Multiple collections**: Organizes memory into specialized collections:
  - `complete_analysis`: Stores full analysis results
  - `visual_elements`: Stores visual component analyses
  - `style_analysis`: Stores style and aesthetic analyses
  - `scenario_analysis`: Stores scenario interpretations

### Memory Operations

- **Exact matching**: Instantly retrieves cached results for identical images via hash lookup
- **Semantic retrieval**: Finds similar past analyses using vector similarity search
- **Memory-enhanced inference**: Includes relevant past analyses when determining context
- **Automatic storage**: Stores each new analysis for future reference and learning

## Architecture

The application consists of four main components:

1. **MCP Server** (`main.py`): Provides analysis tools and memory management
2. **Memory Module** (`modules/memory.py`): Handles storage and retrieval of analyses
3. **Streamlit UI** (`app.py`): Provides a user-friendly interface for uploading images and viewing results
4. **ChromaDB**: Vector database for semantic storage and retrieval of analyses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
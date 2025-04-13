 # Context Detective

Context Detective is an AI-powered application that analyzes images to determine their context, meaning, and significance. It uses a combination of visual analysis, style recognition, and web search to provide comprehensive insights about images.

## Features

- **Visual Elements Analysis**: Identifies objects, people, colors, and text in images
- **Style Analysis**: Recognizes artistic styles and cultural elements
- **Scenario Analysis**: Determines what might be happening in the image
- **Context Inference**: Provides a comprehensive analysis of the image's context
- **Confidence Rating**: Indicates how confident the system is in its analysis

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

3. Create a `.env` file in the root directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Upload an image using the file uploader.

3. Click the "Analyze Image" button to start the analysis.

4. View the results, which include:
   - Context Guess
   - Confidence Rating
   - Explanation
   - Related Links
   - Search Terms Used

## How It Works

Context Detective uses a multi-step workflow to analyze images:

1. **Visual Elements Analysis**: Identifies objects, people, colors, and text in the image.
2. **Style Analysis**: Recognizes artistic styles and cultural elements.
3. **Scenario Analysis**: Determines what might be happening in the image.
4. **Context Inference**: Combines all analyses to provide a comprehensive understanding of the image's context.
5. **Final Output**: Formats the results with a confidence rating and explanation.

## Architecture

The application consists of three main components:

1. **MCP Server** (`example2-detective.py`): Provides tools for image analysis.
2. **Client** (`talk2mcp-detective.py`): Communicates with the MCP server to process images.
3. **Streamlit UI** (`app.py`): Provides a user-friendly interface for uploading images and viewing results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
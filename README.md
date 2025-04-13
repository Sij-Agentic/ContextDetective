# 🧠 Agentic Context Detector

An intelligent system that analyzes images to determine their context, meaning, and significance through multi-stage analysis and web exploration.

## 🎯 Features

- Multi-perspective image analysis
- Web-based evidence gathering
- Context inference with confidence ratings
- Structured output with explanations
- Extensible architecture

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. Run the context detector:
   ```bash
   python context_detective_workflow.py
   ```

## 🧩 How It Works

1. **Image Input**: Accepts one or more input images
2. **Multi-Perspective Analysis**:
   - Visual elements detection
   - Style and aesthetics analysis
   - Scenario inference
3. **Web Exploration**:
   - Generates search terms
   - Gathers supporting evidence
4. **Context Inference**:
   - Combines all analyses
   - Produces confidence-rated conclusions
5. **Structured Output**:
   - Context guess
   - Confidence score
   - Explanation
   - Related links
   - Search terms used

## 🔧 Configuration

- Set your API keys in `.env`
- Adjust analysis parameters in `context_detective.py`
- Modify workflow in `context_detective_workflow.py`

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
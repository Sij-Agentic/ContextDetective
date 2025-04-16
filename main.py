import os
import sys
import json
import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from typing import Dict, Any, List

# modular components
from modules.perception import PerceptionModule
from modules.memory import MemoryModule 
from modules.decision import DecisionModule
from modules.action import ActionModule

# Set up logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"context_detective_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ContextDetective")
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

# Load environment variables
load_dotenv()

# Initialize MCP
mcp = FastMCP("ContextDetective")

# Initialize our modules
perception = PerceptionModule(os.getenv("GEMINI_API_KEY"))
memory = MemoryModule()
decision = DecisionModule(os.getenv("GEMINI_API_KEY"))
action = ActionModule()

# Define the system prompt
SYSTEM_PROMPT = """
You are an agentic context detector that analyzes images to determine their context, meaning, and significance.

## Reasoning Guidelines
1. **Step-by-Step Analysis**: Before providing any output, think aloud in a detailed, logical sequence. Label each type of reasoning (e.g., [Visual Analysis], [Lookup], [Inference]) where appropriate.
2. **Self-Check**: After completing each major reasoning step, briefly review whether your conclusions remain consistent and coherent. If any doubts arise, re-evaluate that step.
3. **Fallback Handling**: If you cannot determine the correct interpretation or if a tool fails, provide a best-guess explanation and clearly state any uncertainties.

## Multi-Turn Conversation Support
- If the conversation continues (e.g., user or system provides more details), integrate that information into your existing reasoning chain.
- Carry forward relevant context from previous steps or turns.

### Tool Usage
- Use the following format to invoke tools (one per line):
  - FUNCTION_CALL: tool_name|param1|param2|...
- Always include 'FUNCTION_CALL:' followed by the tool name and any parameters.
- When you are ready to provide the final structured result, use `FINAL_ANSWER:` followed by the answer.

### Workflow Steps
1. [Perception] Analyze the image using multiple perspectives:
   - Visual elements (objects, people, colors, text)
   - Style and aesthetics (artistic style, cultural elements)
   - Possible scenarios (what might be happening)
   - **Self-check**: Confirm your observed elements align logically.
2. [Decision] Generate search terms based on your findings and explore the web.
   - **Self-check**: Ensure the terms accurately reflect the image details.
3. [Memory] Check if similar images have been analyzed before.
4. [Decision] Infer the most likely context and meaning behind the image.
   - **Self-check**: Verify consistency with earlier steps.
5. [Action] Provide structured output, including your overall confidence rating.
   - Follow the specified response format and label each part of your answer.

### Tools:
{tools_block}

### Response Format
1. **Use FUNCTION_CALL** whenever you need to consult a tool. 
2. Once you have completed all steps and verified your reasoning, return a **FINAL_ANSWER** containing your structured output. 
   - This structured output must reflect the same content as originally intended: the context, meaning, and significance of the image, accompanied by a confidence rating.
"""

# Register perception tools
@mcp.tool()
async def describe_visual_elements(image_path: str) -> Dict[str, Any]:
    """Analyze and describe the visual elements in an image."""
    return await perception.describe_visual_elements(image_path)

@mcp.tool()
async def describe_style_or_aesthetics(image_path: str) -> Dict[str, Any]:
    """Analyze and describe the style and aesthetic features of an image."""
    return await perception.describe_style_or_aesthetics(image_path)

@mcp.tool()
async def describe_possible_scenario(image_path: str) -> Dict[str, Any]:
    """Analyze and describe possible scenarios or contexts for the image."""
    return await perception.describe_possible_scenario(image_path)

# Register memory tools
@mcp.tool()
async def store_analysis(image_hash: str, analysis_json: str) -> Dict[str, Any]:
    """Store the current analysis in memory."""
    return await memory.store_analysis(image_hash, analysis_json)

@mcp.tool()
async def retrieve_similar_analyses(image_hash: str) -> Dict[str, Any]:
    """Retrieve similar previous analyses."""
    return await memory.retrieve_similar_analyses(image_hash)

# Register decision tools
@mcp.tool()
async def generate_search_terms(descriptions: List[str]) -> Dict[str, Any]:
    """Generate relevant search terms from the image descriptions."""
    return await decision.generate_search_terms(descriptions)

@mcp.tool()
async def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information related to the query."""
    return await decision.search_web(query)

@mcp.tool()
async def infer_context(
    visual_elements: str,
    style_analysis: str,
    scenario_analysis: str,
    web_findings: str
) -> Dict[str, Any]:
    """Combine all analyses to infer the most likely context."""
    return await decision.infer_context(
        visual_elements, style_analysis, scenario_analysis, web_findings
    )

# Register action tools
@mcp.tool()
async def format_final_output(
    context_guess: str,
    confidence: float,
    explanation: str,
    related_links: List[str],
    search_terms: List[str]
) -> Dict[str, Any]:
    """Format the final output in the required JSON structure."""
    return await action.format_final_output(
        context_guess, confidence, explanation, related_links, search_terms
    )

@mcp.prompt()
def get_system_prompt() -> str:
    return SYSTEM_PROMPT

# Get user preferences (add this functionality)
def get_user_preferences():
    """Get user preferences before starting the analysis."""
    interests = input("Please enter your interests (comma-separated): ").split(",")
    interests = [i.strip() for i in interests if i.strip()]
    
    if not interests:
        interests = ["general knowledge"]
        print("No interests provided, using 'general knowledge' as default.")
    
    return {"interests": interests}

if __name__ == "__main__":
    print("Context Detective MCP Server Starting...")
    
    # Get user preferences before starting
    user_prefs = get_user_preferences()
    
    # Store preferences in environment for tools to access
    os.environ["USER_PREFERENCES"] = json.dumps(user_prefs)
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        mcp.run(transport="stdio")
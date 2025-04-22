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
import uuid
from mcp.types import TextContent

# modular components
from modules.perception import PerceptionModule
from modules.memory import MemoryModule 
from modules.decision import DecisionModule
from modules.action import ActionModule

# Pydantic models
from models.schemas import (
    PerceptionInput, SearchTermsInput, WebSearchInput, 
    ContextInferenceInput, MemoryStoreInput, MemoryRetrieveInput,
    FinalOutputInput
)

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
            logging.FileHandler(log_file, encoding='utf-8'),
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
memory = MemoryModule(os.getenv("MEMORY_PATH", "memory_storage"), os.getenv("OLLAMA_URL", "http://localhost:11434"))
decision = DecisionModule(os.getenv("GEMINI_API_KEY"))
action = ActionModule()

# Define the system prompt
SYSTEM_PROMPT = """
FOLLOW THESE EXACT INSTRUCTIONS:

1. You must ONLY respond with ONE of these formats:
   FUNCTION_CALL: tool_name|parameter
   FINAL_ANSWER: {"json": "output"}

2. MEMORY-ENHANCED WORKFLOW - FOLLOW THIS SEQUENCE:
   
   A. MEMORY CHECK PHASE
   1️⃣ FUNCTION_CALL: compute_image_hash|C:\\path\\to\\image.png
   2️⃣ FUNCTION_CALL: check_exact_match|image_hash
   * If a match is found, go straight to FINAL_ANSWER
   
   B. ANALYSIS PHASE
   3️⃣ FUNCTION_CALL: describe_visual_elements|C:\\path\\to\\image.png
   4️⃣ FUNCTION_CALL: describe_style_or_aesthetics|C:\\path\\to\\image.png
   5️⃣ FUNCTION_CALL: describe_possible_scenario|C:\\path\\to\\image.png
   
   C. SEARCH & MEMORY RETRIEVAL
   6️⃣ FUNCTION_CALL: generate_search_terms
   7️⃣ FUNCTION_CALL: search_web|search query
   8️⃣ FUNCTION_CALL: retrieve_memory_for_inference
   
   D. INFERENCE & STORAGE
   9️⃣ FUNCTION_CALL: infer_context
   🔟 FUNCTION_CALL: store_analysis|image_hash|json_analysis
   
   E. FINAL RESULT
   1️⃣1️⃣ FINAL_ANSWER: {"json output"}

CRITICAL: FOLLOW THE EXACT SEQUENCE ABOVE. You MUST complete describe_visual_elements, describe_style_or_aesthetics, and describe_possible_scenario BEFORE calling generate_search_terms.

3. DO NOT add ANY explanations before or after your function call.
4. DO NOT use ANY other format or tools than those listed above.
5. DO NOT include "image_path" or any parameter names in your function calls.

Example: FUNCTION_CALL: describe_visual_elements|C:\\Users\\path\\to\\image.png
"""

# Register perception tools
@mcp.tool()
async def describe_visual_elements(input_data: PerceptionInput) -> Dict[str, Any]:
    """Analyze and describe the visual elements in an image."""
    return await perception.describe_visual_elements(input_data.image_path)

@mcp.tool()
async def describe_style_or_aesthetics(input_data: PerceptionInput) -> Dict[str, Any]:
    """Analyze and describe the style and aesthetic features of an image."""
    return await perception.describe_style_or_aesthetics(input_data.image_path)

@mcp.tool()
async def describe_possible_scenario(input_data: PerceptionInput) -> Dict[str, Any]:
    """Analyze and describe possible scenarios or contexts for the image."""
    return await perception.describe_possible_scenario(input_data.image_path)

# Register memory tools
@mcp.tool()
async def store_analysis(input_data: MemoryStoreInput) -> Dict[str, Any]:
    """Store the current analysis in memory."""
    return await memory.store_analysis(input_data.image_hash, input_data.analysis_json)

@mcp.tool()
async def check_exact_match(input_data: MemoryRetrieveInput) -> Dict[str, Any]:
    """Check if we have an exact match for this image hash."""
    return await memory.check_exact_match(input_data.image_hash)

@mcp.tool()
async def retrieve_similar_analyses(input_data: WebSearchInput) -> Dict[str, Any]:
    """Retrieve semantically similar analyses based on text query."""
    return await memory.retrieve_similar_analyses(input_data.query)

@mcp.tool()
async def retrieve_memory_for_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant past analyses to augment inference."""
    return await memory.retrieve_memory_for_inference(input_data)

# Register decision tools
@mcp.tool()
async def generate_search_terms(input_data: SearchTermsInput) -> Dict[str, Any]:
    """Generate relevant search terms from the image descriptions."""
    return await decision.generate_search_terms(input_data.descriptions)

@mcp.tool()
async def search_web(input_data: WebSearchInput) -> Dict[str, Any]:
    """Search the web for information related to the query."""
    return await decision.search_web(input_data.query)

@mcp.tool()
async def infer_context(input_data: ContextInferenceInput) -> Dict[str, Any]:
    """Combine all analyses to infer the most likely context."""
    return await decision.infer_context(
        input_data.visual_elements, 
        input_data.style_analysis, 
        input_data.scenario_analysis, 
        input_data.web_findings
    )

# Register action tools
@mcp.tool()
async def format_final_output(input_data: FinalOutputInput) -> Dict[str, Any]:
    """Format the final output in the required JSON structure."""
    return await action.format_final_output(
        input_data.context_guess, 
        input_data.confidence, 
        input_data.explanation, 
        input_data.related_links, 
        input_data.search_terms
    )

# Add this new tool to compute image hash
@mcp.tool()
async def compute_image_hash(input_data: PerceptionInput) -> Dict[str, Any]:
    """Compute a hash for an image to uniquely identify it."""
    try:
        image_path = input_data.image_path
        logger.info(f"Computing hash for image at path: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
            
        image_hash = memory._compute_image_hash(image_path)
        
        # Make sure we're not returning "Unknown" or any error string
        if image_hash and not image_hash.startswith("error_"):
            logger.info(f"🔑 Successfully computed image hash: {image_hash} for {image_path}")
            return {
                "content": [TextContent(type="text", text=image_hash)]
            }
        else:
            error_msg = f"Failed to compute valid hash, got: {image_hash}"
            logger.error(error_msg)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
    except Exception as e:
        error_msg = f"Error computing image hash: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return {
            "content": [TextContent(type="text", text=f"❌ {error_msg}")]
        }

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
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        # Use TCP transport
        mcp.run(transport="sse") 

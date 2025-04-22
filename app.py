import os
import sys
import json
import logging
import datetime
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import re
import uuid
import streamlit as st
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
# Import MCP components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import Pydantic models
from models.schemas import (
    UserPreferences, ActionOutput, 
    PerceptionInput, SearchTermsInput, WebSearchInput, 
    ContextInferenceInput, MemoryStoreInput, MemoryRetrieveInput,
    FinalOutputInput
)
from mcp.client.sse import sse_client

# Import Gemini
from google import genai

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Context Detective",
    page_icon="🔍",
    layout="wide"
)

# Setup logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_ui = log_dir / f"streamlit_app_{timestamp}.log"
    log_file_client = log_dir / f"mcp_client_{timestamp}.log"
    
    # UI Logger
    ui_logger = logging.getLogger("ContextDetectiveUI")
    ui_logger.setLevel(logging.DEBUG)
    ui_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ui_fh = logging.FileHandler(log_file_ui, encoding='utf-8')
    ui_fh.setFormatter(ui_formatter)
    ui_sh = logging.StreamHandler()
    ui_sh.setFormatter(ui_formatter)
    
    ui_logger.addHandler(ui_fh)
    ui_logger.addHandler(ui_sh)
    ui_logger.info(f"UI Log file created at: {log_file_ui}")

    # Client Logger (logs interactions with MCP server)
    client_logger = logging.getLogger("ContextDetectiveClient")
    client_logger.setLevel(logging.INFO) # Keep INFO for client logs unless debugging MCP
    client_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    client_fh = logging.FileHandler(log_file_client, encoding='utf-8')
    client_fh.setFormatter(client_formatter)
    
    client_logger.addHandler(client_fh)
    client_logger.info(f"MCP Client Log file created at: {log_file_client}")

    return ui_logger, client_logger

logger, client_logger = setup_logging()

# Add this function near the beginning of the file, after imports
def extract_hash_from_response(response_text):
    """Extract the actual hash value from a potentially nested JSON response."""
    try:
        # First check if it's a JSON string
        if response_text.startswith('{') and response_text.endswith('}'):
            json_data = json.loads(response_text)
            
            # Handle nested content array structure
            if 'content' in json_data and isinstance(json_data['content'], list):
                for item in json_data['content']:
                    if 'text' in item:
                        return item['text']
            
            # If not found in the expected structure, search recursively
            def find_hash(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'text' and isinstance(value, str) and len(value) == 32:
                            return value
                        result = find_hash(value)
                        if result:
                            return result
                elif isinstance(obj, list):
                    for item in obj:
                        result = find_hash(item)
                        if result:
                            return result
                return None
            
            return find_hash(json_data)
        else:
            # Already a simple string
            return response_text
    except:
        # If parsing fails, return original
        return response_text

# --- MCP Client Workflow State ---
class WorkflowState:
    def __init__(self):
        self.iteration = 0
        self.visual_elements = None
        self.style_analysis = None
        self.scenario_analysis = None
        self.search_terms = None
        self.web_findings = None
        self.context_inference = None
        self.final_output = None
        self.image_path = None
        self.errors = []
        self.analysis_log = [] # Store log messages for UI display
        self.session_id = None
        self.memory_context = None

    def log_step(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.analysis_log.append(log_entry)
        client_logger.info(message) # Also log to file

    def update(self, key: str, value: Any):
        setattr(self, key, value)
        self.log_step(f"State Updated: {key} set.")
        # Log truncated value for debugging large outputs
        if isinstance(value, str) and len(value) > 200:
            client_logger.debug(f"{key} value (truncated): {value[:100]}...{value[-100:]}")
        else:
            client_logger.debug(f"{key} value: {value}")

    def get_progress_text(self) -> str:
        # Simple text progress based on collected logs
        return "\n".join(self.analysis_log)

# --- Gemini Interaction ---
def generate_agent_step(
    client: ClientSession,
    image_path: str,
    analysis_state: dict,
    history: List[str]
) -> str:
    """Generate the next step for the context detection agent."""
    
    visual_status = "Visual elements analysis is available." if "visual_elements" in analysis_state and analysis_state["visual_elements"] else "Visual elements have not been analyzed yet. Consider using describe_visual_elements next."

    style_status = "Style/aesthetics analysis is available." if "style_or_aesthetics" in analysis_state and analysis_state["style_or_aesthetics"] else "Style/aesthetics have not been analyzed yet. Consider using describe_style_or_aesthetics after visual elements."
    
    scenario_status = "Scenario analysis is available." if "possible_scenario" in analysis_state and analysis_state["possible_scenario"] else "Possible scenarios have not been analyzed yet. Consider using describe_possible_scenario after style analysis."
    
    search_terms_status = "Search terms have been generated." if "search_terms" in analysis_state and analysis_state["search_terms"] else "Search terms have not been generated yet. Consider using generate_search_terms after all analyses are complete."
    
    web_search_status = "Web search has been performed." if "search_results" in analysis_state and analysis_state["search_results"] else "Web search has not been performed yet. Consider using search_web after generating search terms."
    
    context_status = "Context has been inferred." if "inferred_context" in analysis_state and analysis_state["inferred_context"] else "Context has not been inferred yet. Consider using infer_context after web search."
    
    prompt = f"""
You are ContextDetective, an agent for analyzing images to determine historical or cultural contexts.

Current image path: {image_path}
Current analysis state: {json.dumps(analysis_state, indent=2)}
Previous steps: {json.dumps(history, indent=2)}

Analysis Status:
- {visual_status}
- {style_status}
- {scenario_status}
- {search_terms_status}
- {web_search_status}
- {context_status}

Determine the appropriate next step in the analysis.
"""
    
    format_reminder = """
CONTEXT DETECTIVE WORKFLOW INSTRUCTIONS:

## Goal
You are analyzing images to determine their historical or cultural context by following a structured workflow.

## Reasoning and Tool Use Process
For each step in your analysis:
1. First, briefly explain your reasoning (1-2 sentences)
2. Then use the appropriate tool by writing: FUNCTION_CALL: tool_name|parameter
3. After receiving results, verify if they make sense before proceeding

## Required Analysis Sequence
You MUST follow this exact sequence:
1️⃣ FUNCTION_CALL: describe_visual_elements|[image_path]
2️⃣ FUNCTION_CALL: describe_style_or_aesthetics|[image_path]
3️⃣ FUNCTION_CALL: describe_possible_scenario|[image_path]
4️⃣ FUNCTION_CALL: generate_search_terms
5️⃣ FUNCTION_CALL: search_web|[search query]
6️⃣ FUNCTION_CALL: infer_context
7️⃣ FINAL_ANSWER: [structured JSON]

## Output Rules
- Your FUNCTION_CALL should use pipe delimiter format: tool_name|parameter
- For image analysis tools, use the exact image path without parameter names
- Your FINAL_ANSWER must use this JSON structure:
  {
    "context_guess": "Brief description of the historical/cultural context",
    "confidence": 0.75,
    "explanation": "Detailed explanation of why this context is likely",
    "related_links": ["link1", "link2"],
    "search_terms_used": ["term1", "term2"]
  }

## Error Handling
- If uncertain about a tool's output, note your concerns before proceeding
- If a tool seems to fail, explain the issue and retry or proceed with caution
- Include related_links even if few were found; use empty array if none

## Examples
Good function call with reasoning:
"I need to identify the visual elements first. FUNCTION_CALL: describe_visual_elements|C:\\path\\to\\image.png"

Good final answer:
"Based on the analysis, I can now provide a structured conclusion. FINAL_ANSWER: {\"context_guess\":\"Japanese Ukiyo-e art from Edo period\",\"confidence\":0.85,\"explanation\":\"The visual elements and style match Ukiyo-e woodblock prints\",\"related_links\":[\"https://example.com/ukiyo-e\"],\"search_terms_used\":[\"woodblock prints\",\"japanese art\"]}"
"""

    full_prompt = prompt + "\n\n" + format_reminder
    
    logging.info(f"Generated agent prompt:\n{full_prompt}")
    
    # Create Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    try:
        # Call the Gemini API with text-only model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[full_prompt],
        )
        
        result = response.text
        logging.info(f"Raw API response: {result}")
        
        return result
    except Exception as e:
        logging.error(f"Error calling Gemini API: {str(e)}")
        return f"FUNCTION_CALL: describe_visual_elements|{image_path}"

# --- Tool Formatting and Execution ---
def format_tools_for_prompt(tools: List[Any]) -> str:
    """Format tools information for the system prompt."""
    tool_descriptions = []
    for i, tool in enumerate(tools):
        try:
            params_str = str(tool.inputSchema.get("properties", "No parameters")) 
            desc = getattr(tool, 'description', 'No description')
            name = getattr(tool, 'name', f'tool_{i}')
            tool_descriptions.append(f"{i+1}. {name}({params_str}) - {desc}")
        except Exception as e:
            client_logger.error(f"Error formatting tool {i}: {str(e)}")
            tool_descriptions.append(f"{i+1}. Error loading tool details: {str(e)}")
    return "\n".join(tool_descriptions)

async def execute_tool_call(
    session: ClientSession, 
    tool_name: str, 
    params: List[str], 
    state: WorkflowState
) -> Optional[str]:
    """Execute a single tool call via MCP session."""
    state.log_step(f"Executing tool: {tool_name} with params: {params}")
    
    # Define the allowed tools in the workflow
    allowed_tools = [
        "describe_visual_elements", 
        "describe_style_or_aesthetics",
        "describe_possible_scenario",
        "generate_search_terms",
        "search_web",
        "infer_context",
        "format_final_output",
        "store_analysis",
        "retrieve_similar_analyses"
    ]
    
    workflow_sequence = [
        "describe_visual_elements",
        "describe_style_or_aesthetics", 
        "describe_possible_scenario",
        "generate_search_terms",
        "search_web",
        "infer_context",
        "format_final_output"
    ]
    
    # Check if the tool is allowed
    if tool_name not in allowed_tools:
        error_message = f"Tool '{tool_name}' is not in the allowed workflow tools. Please use only: {', '.join(allowed_tools)}"
        state.log_step(f"❌ {error_message}")
        state.errors.append(error_message)
        return f"Error: {error_message}"
    
    if tool_name == "describe_visual_elements" and state.visual_elements:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.visual_elements
    elif tool_name == "describe_style_or_aesthetics" and state.style_analysis:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.style_analysis
    elif tool_name == "describe_possible_scenario" and state.scenario_analysis:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.scenario_analysis
    elif tool_name == "generate_search_terms" and state.search_terms:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.search_terms
    elif tool_name == "search_web" and state.web_findings:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.web_findings
    elif tool_name == "infer_context" and state.context_inference:
        state.log_step(f"⚠️ Tool {tool_name} has already been executed. Using cached results.")
        return state.context_inference
    
    if tool_name in workflow_sequence:
        current_index = workflow_sequence.index(tool_name)
        
        for i in range(current_index):
            previous_tool = workflow_sequence[i]
            
            if previous_tool == "describe_visual_elements" and not state.visual_elements:
                error_message = f"Cannot call {tool_name} yet. You must first call describe_visual_elements."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
                
            elif previous_tool == "describe_style_or_aesthetics" and not state.style_analysis:
                error_message = f"Cannot call {tool_name} yet. You must first call describe_style_or_aesthetics."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
                
            elif previous_tool == "describe_possible_scenario" and not state.scenario_analysis:
                error_message = f"Cannot call {tool_name} yet. You must first call describe_possible_scenario."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
                
            elif previous_tool == "generate_search_terms" and not state.search_terms and tool_name not in ["describe_visual_elements", "describe_style_or_aesthetics", "describe_possible_scenario"]:
                error_message = f"Cannot call {tool_name} yet. You must first call generate_search_terms."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
                
            elif previous_tool == "search_web" and not state.web_findings and tool_name not in ["describe_visual_elements", "describe_style_or_aesthetics", "describe_possible_scenario", "generate_search_terms"]:
                error_message = f"Cannot call {tool_name} yet. You must first call search_web."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
                
            elif previous_tool == "infer_context" and not state.context_inference and tool_name == "format_final_output":
                error_message = f"Cannot call {tool_name} yet. You must first call infer_context."
                state.log_step(f"❌ {error_message}")
                state.errors.append(error_message)
                return f"Error: {error_message}. Follow the required sequence."
    
    try:
        normalized_params = []
        for p in params:
            if "image_path" in p and ("C:" in p or "/" in p or "\\" in p):
                if "image_path=" in p:
                    path_value = p.split("image_path=", 1)[1].strip()
                    normalized_params.append(path_value)
                    state.log_step(f"Extracted path from image_path= format: {path_value}")
                elif "image_path:" in p:
                    path_value = p.split("image_path:", 1)[1].strip()
                    normalized_params.append(path_value)
                    state.log_step(f"Extracted path from image_path: format: {path_value}")
                else:
                    normalized_params.append(p.strip())
            elif "=" in p and not p.startswith("http"):
                normalized_params.append(p.split("=", 1)[1].strip())
            elif ":" in p and not (p.startswith("http") or p.startswith("C:\\") or p.startswith("c:\\")):
                normalized_params.append(p.split(":", 1)[1].strip())
            else:
                param = p.strip()
                if param.startswith('"') and param.endswith('"'):
                    param = param[1:-1]  # Remove surrounding quotes
                normalized_params.append(param)
        
        state.log_step(f"Normalized parameters: {normalized_params}")
        
        if tool_name == "describe_visual_elements" or tool_name == "describe_style_or_aesthetics" or tool_name == "describe_possible_scenario":
            if normalized_params:
                input_model = PerceptionInput(image_path=normalized_params[-1])
            else:
                state.log_step(f"❌ Error: No image path provided for {tool_name}")
                state.errors.append(f"TOOL ERROR: Tool {tool_name} requires an image path. Use the exact format: FUNCTION_CALL: {tool_name}|C:\\path\\to\\image.png")
                return f"Error: No image path provided for {tool_name}. Correct format: {tool_name}|image_path"
                
        elif tool_name == "generate_search_terms":
            descriptions = []
            if state.visual_elements:
                descriptions.append(state.visual_elements)
                state.log_step(f"DEBUG: Using visual elements for search terms: {state.visual_elements[:100]}...")
            if state.style_analysis:
                descriptions.append(state.style_analysis)
                state.log_step(f"DEBUG: Using style analysis for search terms: {state.style_analysis[:100]}...")
            if state.scenario_analysis:
                descriptions.append(state.scenario_analysis)
                state.log_step(f"DEBUG: Using scenario analysis for search terms: {state.scenario_analysis[:100]}...")
            
            if descriptions:
                state.log_step(f"DEBUG: Calling generate_search_terms with {len(descriptions)} descriptions")
                search_input = {"descriptions": descriptions}
                state.log_step(f"DEBUG: Search input created")
                
                search_terms_result = await session.call_tool("generate_search_terms", 
                                                       arguments={"input_data": search_input})
                
                state.log_step(f"DEBUG: Got search terms result: {search_terms_result}")
                
                if hasattr(search_terms_result, 'content') and search_terms_result.content:
                    search_terms_text = search_terms_result.content[0].text
                    state.search_terms = search_terms_text
                    state.log_step(f"✅ Search terms generation complete: {search_terms_text}")
                else:
                    state.log_step("⚠️ Search terms result has no content")
            else:
                state.log_step("⚠️ No descriptions available for search terms generation")
                
        elif tool_name == "search_web":
            if not normalized_params:
                state.log_step(f"❌ Error: No query provided for search_web")
                state.errors.append(f"TOOL ERROR: Tool search_web requires a query. Use the exact format: FUNCTION_CALL: search_web|your search query")
                return "Error: No query provided for search_web. Correct format: search_web|your search query"
            input_model = WebSearchInput(query=normalized_params[0])
            
        elif tool_name == "infer_context":
            # Check if state has required data
            if not state.visual_elements or not state.style_analysis or not state.scenario_analysis:
                error_msg = "Cannot execute infer_context yet - missing required analyses"
                state.log_step(f"❌ {error_msg}")
                state.errors.append(f"TOOL ERROR: {error_msg}. Complete all visual analysis steps first.")
                return f"Error: {error_msg}. Complete visual_elements, style_analysis, and scenario_analysis first."
            
            # Extract actual text content from the state values
            visual_elements_text = state.visual_elements
            if isinstance(visual_elements_text, dict) and 'content' in visual_elements_text:
                try:
                    visual_elements_text = '\n'.join([item.get('text', '') for item in visual_elements_text['content'] if item.get('text')])
                except:
                    visual_elements_text = str(visual_elements_text)
                    
            style_analysis_text = state.style_analysis
            if isinstance(style_analysis_text, dict) and 'content' in style_analysis_text:
                try:
                    style_analysis_text = '\n'.join([item.get('text', '') for item in style_analysis_text['content'] if item.get('text')])
                except:
                    style_analysis_text = str(style_analysis_text)
                    
            scenario_analysis_text = state.scenario_analysis
            if isinstance(scenario_analysis_text, dict) and 'content' in scenario_analysis_text:
                try:
                    scenario_analysis_text = '\n'.join([item.get('text', '') for item in scenario_analysis_text['content'] if item.get('text')])
                except:
                    scenario_analysis_text = str(scenario_analysis_text)
                    
            web_findings_text = state.web_findings if state.web_findings else "No web findings available."
            if isinstance(web_findings_text, dict) and 'content' in web_findings_text:
                try:
                    web_findings_text = '\n'.join([item.get('text', '') for item in web_findings_text['content'] if item.get('text')])
                except:
                    web_findings_text = str(web_findings_text)
            
            state.log_step("Extracted text content from all analyses for infer_context")
            
            input_model = ContextInferenceInput(
                visual_elements=visual_elements_text,
                style_analysis=style_analysis_text,
                scenario_analysis=scenario_analysis_text,
                web_findings=web_findings_text
            )
            
        elif tool_name == "store_analysis":
            if len(normalized_params) < 2:
                state.log_step(f"❌ Error: Missing parameters for store_analysis")
                state.errors.append(f"TOOL ERROR: Tool store_analysis requires image_hash and analysis_json. Use the exact format: FUNCTION_CALL: store_analysis|hash|json_data")
                return "Error: store_analysis requires image_hash and analysis_json. Correct format: store_analysis|hash|json_data"
            input_model = MemoryStoreInput(
                image_hash=normalized_params[0],
                analysis_json=normalized_params[1]
            )
            
        elif tool_name == "retrieve_similar_analyses":
            if not normalized_params:
                state.log_step(f"❌ Error: No image_hash provided for retrieve_similar_analyses")
                state.errors.append(f"TOOL ERROR: Tool retrieve_similar_analyses requires an image_hash. Use the exact format: FUNCTION_CALL: retrieve_similar_analyses|hash")
                return "Error: No image_hash provided for retrieve_similar_analyses. Correct format: retrieve_similar_analyses|hash"
            input_model = MemoryRetrieveInput(image_hash=normalized_params[0])
            
        elif tool_name == "format_final_output":
            try:
                if len(normalized_params) < 3:
                    state.log_step(f"❌ Error: Insufficient parameters for format_final_output")
                    state.errors.append(f"TOOL ERROR: Tool format_final_output requires at least context_guess, confidence, explanation. Use the exact format: FUNCTION_CALL: format_final_output|context|0.8|explanation|links|terms")
                    return "Error: format_final_output requires context_guess, confidence, explanation. Correct format: format_final_output|context|0.8|explanation|['link1']|['term1']"
                
                # Parse confidence value
                try:
                    confidence = float(normalized_params[1])
                except ValueError:
                    state.log_step(f"Warning: Invalid confidence value '{normalized_params[1]}', using 0.5")
                    confidence = 0.5
                
                # Parse lists
                related_links = json.loads(normalized_params[3]) if len(normalized_params) > 3 and normalized_params[3].startswith('[') else [normalized_params[3] if len(normalized_params) > 3 else ""]
                search_terms = json.loads(normalized_params[4]) if len(normalized_params) > 4 and normalized_params[4].startswith('[') else [normalized_params[4] if len(normalized_params) > 4 else ""]
                
                input_model = FinalOutputInput(
                    context_guess=normalized_params[0],
                    confidence=confidence,
                    explanation=normalized_params[2],
                    related_links=related_links,
                    search_terms=search_terms
                )
            except Exception as e:
                state.log_step(f"❌ Error parsing format_final_output parameters: {str(e)}")
                state.errors.append(f"TOOL ERROR: Failed to parse format_final_output parameters: {str(e)}. Check parameter format.")
                return f"Error parsing format_final_output parameters: {str(e)}. Correct format: format_final_output|context|0.8|explanation|['link1']|['term1']"
        else:
            # For any other tools, just pass through the parameters as a dict
            state.log_step(f"Warning: No input model defined for {tool_name}, passing parameters as-is")
            input_model = {f"param{i}": p for i, p in enumerate(normalized_params)}

        # Execute the tool with properly structured input
        client_logger.info(f"Calling MCP tool '{tool_name}' with input model: {input_model}")
        
        if hasattr(input_model, "dict"):
            arguments_dict = {"input_data": input_model.dict()}
            result = await session.call_tool(tool_name, arguments=arguments_dict)
        else:
            result = await session.call_tool(tool_name, arguments={"input_data": input_model})
            
        client_logger.info(f"Received result from tool '{tool_name}'")

        # Extract text content from result
        texts = []
        if hasattr(result, 'content'):
            for item in result.content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                else:
                    texts.append(str(item)) 
        else:
            texts.append(str(result))  # Fallback

        result_str = "\n".join(texts)
        state.log_step(f"Tool '{tool_name}' executed.")
        client_logger.debug(f"Tool Result Text (truncated): {result_str[:200]}...")
        
        if result:
            try:
                debug_str = str(result)
                debug_output = debug_str[:200] + "..." if len(debug_str) > 200 else debug_str
            except:
                debug_output = f"<CallToolResult object - cannot display>"
            state.log_step(f"DEBUG - Tool result: {debug_output}")
        
        return result_str
        
    except Exception as e:
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        state.log_step(f"❌ {error_msg}")
        client_logger.error(error_msg, exc_info=True)
        state.errors.append(f"TOOL ERROR: {error_msg}. Check tool parameters and format.")
        return f"Error: {error_msg}. Please check parameters and try again."


async def run_context_analysis(
    image_path: str,
    user_prefs: Dict[str, Any],
    progress_placeholder: "st.delta_generator.DeltaGenerator",
) -> Dict[str, Any]:
    """Runs the full context analysis workflow using MCP with SSE transport."""
    
    # Initialize state first
    state = WorkflowState()
    state.log_step("Starting context analysis...")
    progress_placeholder.text(state.get_progress_text())
    
    # Create a unique session ID for short-term memory tracking
    session_id = str(uuid.uuid4())
    state.session_id = session_id  # Store in state object
    state.log_step(f"Created session ID: {session_id[:8]} for analysis")
    
    # Store image path in state for use in suggestions
    state.image_path = image_path

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables.")
        return None

    # Add this debugging line in run_context_analysis function, right after it receives user_prefs
    state.log_step(f"User interests: {json.dumps(user_prefs)}")
    
    # Make sure the environment variable is properly set:
    os.environ["USER_PREFERENCES"] = json.dumps(user_prefs)
    state.log_step(f"Set USER_PREFERENCES env var: {os.environ.get('USER_PREFERENCES')}")
    
    # First let's compute the image hash - move this after establishing connection
    final_result_json = None
    image_hash = None

    try:
        # Connect to SSE server
        async with sse_client("http://127.0.0.1:8000/sse") as (reader, writer):
            state.log_step("SSE Connection established.")
            progress_placeholder.text(state.get_progress_text())
            
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                state.log_step("MCP Session initialized.")
                progress_placeholder.text(state.get_progress_text())

                # Import memory reference for direct access
                try:
                    from main import memory
                    
                    # First compute hash and check for matches
                    try:
                        # Use the compute_image_hash tool
                        hash_result = await session.call_tool("compute_image_hash", 
                                                           arguments={"input_data": {"image_path": image_path}})
                        
                        # Debug code to see what's coming back
                        state.log_step(f"DEBUG: Raw hash result: {hash_result}")
                        
                        # Extra check for what's in the response
                        if hasattr(hash_result, 'content'):
                            for i, content_item in enumerate(hash_result.content):
                                state.log_step(f"DEBUG: Content item {i}: {content_item}")
                                if hasattr(content_item, 'text'):
                                    state.log_step(f"DEBUG: Content text {i}: {content_item.text}")
                            
                            # Try different methods to extract the hash
                            if hash_result.content:
                                # Try first approach
                                try:
                                    # Get the raw text response
                                    raw_response = hash_result.content[0].text
                                    state.log_step(f"DEBUG: Raw hash text: {raw_response}")
                                    
                                    # Extract the actual hash from nested JSON if needed
                                    image_hash = extract_hash_from_response(raw_response)
                                    state.log_step(f"DEBUG: Extracted hash: {image_hash}")
                                    
                                    if image_hash and image_hash != "Unknown" and not image_hash.startswith("{"):
                                        state.log_step(f"🔑 Computed image hash: {image_hash} for image analysis")
                                    else:
                                        # Fallback to direct hash computation
                                        try:
                                            image_hash = memory._compute_image_hash(image_path)
                                            state.log_step(f"DEBUG: Direct hash computation: {image_hash}")
                                        except Exception as direct_error:
                                            state.log_step(f"DEBUG: Direct hash error: {str(direct_error)}")
                                except Exception as e:
                                    state.log_step(f"DEBUG: Hash extraction error: {str(e)}")
                    except Exception as e:
                        state.log_step(f"⚠️ Failed to check image hash: {str(e)}")
                        # Continue with analysis even if hash check fails
                
                except ImportError as e:
                    # Handle the case where we can't import memory module
                    state.log_step(f"⚠️ Could not import memory module: {str(e)}")
                    memory = None

                # Add this after computing the image hash but before starting the analysis
                if image_hash:
                    # Check for exact matches in memory
                    state.log_step("🔍 Checking for exact hash match in ChromaDB...")
                    match_result = await session.call_tool("check_exact_match", 
                                                         arguments={"input_data": {"image_hash": image_hash}})
                    
                    if hasattr(match_result, 'content') and match_result.content:
                        match_content = match_result.content[0].text
                        if "match_found" in match_content and "true" in match_content.lower():
                            state.log_step("🎯 EXACT MATCH FOUND IN CHROMADB! Hash-based retrieval successful")
                            # Try to parse the cached analysis
                            try:
                                match_data = json.loads(match_content)
                                if 'data' in match_data:
                                    context = match_data['data'].get('context_guess', 'Unknown context')
                                    confidence = match_data['data'].get('confidence', 'Unknown confidence')
                                    state.log_step(f"📋 Retrieved context: {context} (confidence: {confidence})")
                                    state.final_output = match_data['data']
                                    state.log_step("✅ Successfully loaded cached vector analysis from ChromaDB")
                                    return {
                                        "raw_output": state.get_progress_text(),
                                        "structured": state.final_output,
                                        "error": None
                                    }
                            except:
                                state.log_step("⚠️ Failed to parse cached result from ChromaDB, continuing with new analysis")
                        else:
                            state.log_step("ℹ️ No exact hash match in ChromaDB, proceeding with full analysis")

                # Add this right before the infer_context step
                state.log_step("🧠 Retrieving similar analyses through vector search in ChromaDB...")
                try:
                    memory_query = {
                        "visual_elements": state.visual_elements or "",
                        "style_analysis": state.style_analysis or "",
                        "scenario_analysis": state.scenario_analysis or ""
                    }
                    state.log_step(f"📊 Creating vector embedding from current analysis data...")
                    memory_result = await session.call_tool("retrieve_memory_for_inference", 
                                                         arguments={"input_data": memory_query})
                    
                    if hasattr(memory_result, 'content') and memory_result.content:
                        memory_context = memory_result.content[0].text
                        state.memory_context = memory_context
                        
                        # Add clearer logging about vector search
                        num_analyses = memory_context.count('Analysis')
                        if num_analyses > 0:
                            state.log_step(f"🔍 VECTOR SEARCH: Found {num_analyses} semantically similar analyses in ChromaDB")
                            
                            # Try to extract similarity scores
                            similarity_matches = re.findall(r'Similarity: (0\.\d+)', memory_context)
                            if similarity_matches:
                                state.log_step(f"📈 Similarity scores: {', '.join(similarity_matches)}")
                        else:
                            state.log_step("⚠️ VECTOR SEARCH: No semantically similar analyses found in ChromaDB")
                    else:
                        state.log_step("⚠️ VECTOR SEARCH: Query returned empty result from ChromaDB")
                except Exception as e:
                    state.log_step(f"❌ VECTOR SEARCH ERROR: {str(e)}")

                # NOW ADD THE ACTUAL ANALYSIS WORKFLOW
                state.log_step("Starting image analysis workflow...")
                
                # Step 1: Describe visual elements
                state.log_step("Analyzing visual elements...")
                visual_result = await session.call_tool("describe_visual_elements", 
                                                     arguments={"input_data": {"image_path": image_path}})
                if hasattr(visual_result, 'content') and visual_result.content:
                    visual_text = visual_result.content[0].text
                    state.visual_elements = visual_text
                    state.log_step("✅ Visual elements analysis complete")
                    
                    # Store in short-term memory
                    if memory:
                        try:
                            memory.store_in_short_term(session_id, "visual_elements", visual_text)
                            state.log_step("📝 Stored in short-term memory")
                        except Exception as e:
                            state.log_step(f"⚠️ Failed to store in memory: {e}")
                
                # Step 2: Describe style/aesthetics
                state.log_step("Analyzing style and aesthetics...")
                style_result = await session.call_tool("describe_style_or_aesthetics", 
                                                    arguments={"input_data": {"image_path": image_path}})
                if hasattr(style_result, 'content') and style_result.content:
                    style_text = style_result.content[0].text
                    state.style_analysis = style_text
                    state.log_step("✅ Style analysis complete")
                
                # Step 3: Describe possible scenario
                state.log_step("Analyzing possible scenarios...")
                scenario_result = await session.call_tool("describe_possible_scenario", 
                                                       arguments={"input_data": {"image_path": image_path}})
                if hasattr(scenario_result, 'content') and scenario_result.content:
                    scenario_text = scenario_result.content[0].text
                    state.scenario_analysis = scenario_text
                    state.log_step("✅ Scenario analysis complete")
                
                # Step 4: Generate search terms
                state.log_step("Generating search terms based on analyses...")
                try:
                    descriptions = []
                    if state.visual_elements:
                        descriptions.append(state.visual_elements)
                        state.log_step(f"DEBUG: Using visual elements for search terms: {state.visual_elements[:100]}...")
                    if state.style_analysis:
                        descriptions.append(state.style_analysis)
                        state.log_step(f"DEBUG: Using style analysis for search terms: {state.style_analysis[:100]}...")
                    if state.scenario_analysis:
                        descriptions.append(state.scenario_analysis)
                        state.log_step(f"DEBUG: Using scenario analysis for search terms: {state.scenario_analysis[:100]}...")
                    
                    if descriptions:
                        state.log_step(f"DEBUG: Calling generate_search_terms with {len(descriptions)} descriptions")
                        search_input = {"descriptions": descriptions}
                        state.log_step(f"DEBUG: Search input created")
                        
                        search_terms_result = await session.call_tool("generate_search_terms", 
                                                               arguments={"input_data": search_input})
                        
                        state.log_step(f"DEBUG: Got search terms result: {search_terms_result}")
                        
                        if hasattr(search_terms_result, 'content') and search_terms_result.content:
                            search_terms_text = search_terms_result.content[0].text
                            state.search_terms = search_terms_text
                            state.log_step(f"✅ Search terms generation complete: {search_terms_text}")
                        else:
                            state.log_step("⚠️ Search terms result has no content")
                    else:
                        state.log_step("⚠️ No descriptions available for search terms generation")
                    
                except Exception as e:
                    state.log_step(f"❌ Error generating search terms: {str(e)}")
                    # Create a fallback search term based on visual elements
                    if state.visual_elements:
                        state.search_terms = state.visual_elements.split('.')[0] if '.' in state.visual_elements else state.visual_elements[:50]
                        state.log_step(f"⚠️ Using fallback search term: {state.search_terms}")
                
                # Step 5: Search web
                state.log_step("Performing web search for context information...")
                try:
                    if state.search_terms:
                        # Simplify search query extraction
                        search_query = state.search_terms
                        
                        # Clean up the query if it's JSON or has formatting issues
                        try:
                            if search_query.startswith('{') or search_query.startswith('['):
                                # Try to parse JSON
                                search_data = json.loads(search_query)
                                
                                # If it's a list, use the first item
                                if isinstance(search_data, list) and len(search_data) > 0:
                                    search_query = search_data[0]
                                
                                # If it's a dict with terms, use the first term
                                elif isinstance(search_data, dict) and 'terms' in search_data:
                                    if isinstance(search_data['terms'], list) and search_data['terms']:
                                        search_query = search_data['terms'][0]
                                    else:
                                        search_query = str(search_data['terms'])
                        except:
                            # If parsing fails just use the raw text and take the first 100 chars
                            if len(search_query) > 100:
                                search_query = search_query[:100]
                        
                        state.log_step(f"🔍 Using search query: {search_query}")
                        
                        search_input = {"query": search_query}
                        state.log_step(f"DEBUG: Search web input: {search_input}")
                        
                        web_result = await session.call_tool("search_web", arguments={"input_data": search_input})
                        
                        state.log_step(f"DEBUG: Search web result received: {type(web_result)}")
                        
                        if hasattr(web_result, 'content') and web_result.content:
                            state.log_step(f"DEBUG: Content found in web result")
                            web_text = web_result.content[0].text
                            state.web_findings = web_text
                            state.log_step("✅ Web search complete")
                            state.log_step(f"DEBUG: Web findings snippet: {web_text[:150]}...")
                        else:
                            state.log_step("⚠️ Web search returned no content")
                    else:
                        state.log_step("⚠️ No search terms available for web search")
                except Exception as e:
                    state.log_step(f"❌ Error searching web: {str(e)}")
                    state.web_findings = "No web search results due to error."
                
                # Step 6: Infer context with better error handling
                state.log_step("Inferring context from all gathered information...")
                try:
                    # Modify the infer_context input to include memory context
                    infer_input = {
                        "visual_elements": state.visual_elements or "No visual elements analysis available.",
                        "style_analysis": state.style_analysis or "No style analysis available.",
                        "scenario_analysis": state.scenario_analysis or "No scenario analysis available.",
                        "web_findings": state.web_findings or "No web findings available."
                    }
                    
                    # Add memory context if available
                    if state.memory_context:
                        infer_input["memory_context"] = state.memory_context
                        state.log_step("✅ Including memory context in inference")
                    
                    state.log_step(f"DEBUG: Calling infer_context with input: {str(infer_input)[:200]}...")
                    
                    infer_result = await session.call_tool("infer_context", arguments={"input_data": infer_input})
                    
                    state.log_step(f"DEBUG: Infer context result received: {type(infer_result)}")
                    
                    if hasattr(infer_result, 'content') and infer_result.content:
                        state.log_step(f"DEBUG: Content found in inference result")
                        inference_text = infer_result.content[0].text
                        state.context_inference = inference_text
                        state.log_step("✅ Context inference complete")
                        state.log_step(f"DEBUG: Inference snippet: {inference_text[:150]}...")
                    else:
                        state.log_step("⚠️ Inference returned no content")
                except Exception as e:
                    state.log_step(f"❌ Error inferring context: {str(e)}")
                    # Create a simple fallback inference
                    state.context_inference = f"Based on the visual analysis, this appears to be {state.visual_elements[:50] if state.visual_elements else 'an unidentified context'}."
                
                # Step 7: Format final output
                state.log_step("Formatting final output...")
                try:
                    if state.context_inference:
                        state.log_step(f"DEBUG: Using context inference: {state.context_inference[:100]}...")
                        
                        # Try to parse the inference to get structured data
                        context_data = {}
                        search_terms_list = []
                        
                        # Extract search terms used
                        if state.search_terms:
                            state.log_step(f"DEBUG: Extracting from search terms: {state.search_terms[:100]}...")
                            try:
                                if isinstance(state.search_terms, str) and (state.search_terms.startswith("[") or state.search_terms.startswith("{")):
                                    parsed_terms = json.loads(state.search_terms)
                                    state.log_step(f"DEBUG: Parsed search terms JSON successfully")
                                    if isinstance(parsed_terms, list):
                                        search_terms_list = parsed_terms
                                    elif isinstance(parsed_terms, dict) and 'terms' in parsed_terms:
                                        search_terms_list = parsed_terms['terms']
                                else:
                                    search_terms_list = [state.search_terms]
                                    
                                state.log_step(f"DEBUG: Final search terms list: {search_terms_list}")
                            except Exception as e:
                                state.log_step(f"DEBUG: Error parsing search terms: {str(e)}")
                                search_terms_list = [state.search_terms]
                        else:
                            state.log_step("DEBUG: No search terms available")
                            search_terms_list = ["unspecified search terms"]
                        
                        # Try to extract context guess and confidence from inference
                        try:
                            if state.context_inference and isinstance(state.context_inference, str) and state.context_inference.strip().startswith("{"):
                                state.log_step("DEBUG: Attempting to parse JSON from inference")
                                parsed_data = json.loads(state.context_inference)
                                state.log_step(f"DEBUG: JSON parsing successful with keys: {list(parsed_data.keys())}")
                                
                                # Handle the case where we have a 'content' key at the top level
                                if 'content' in parsed_data:
                                    state.log_step(f"DEBUG: Found content key, extracting actual context data")
                                    
                                    # Try to extract content from the nested structure
                                    if isinstance(parsed_data['content'], list) and parsed_data['content']:
                                        # Try to get the actual context data from the first content item
                                        content_item = parsed_data['content'][0]
                                        state.log_step(f"DEBUG: Content item: {content_item}")
                                        
                                        # Extract text if it's there
                                        if isinstance(content_item, dict) and 'text' in content_item:
                                            actual_text = content_item['text']
                                            state.log_step(f"DEBUG: Found text in content: {actual_text[:100]}...")
                                            
                                            # Try to parse the text as JSON if possible
                                            try:
                                                if actual_text.strip().startswith("{"):
                                                    actual_context = json.loads(actual_text)
                                                    state.log_step(f"DEBUG: Successfully parsed nested JSON from text")
                                                    context_data = actual_context
                                                else:
                                                    # Use the text directly with default structure
                                                    context_data = {
                                                        "context_guess": actual_text.split('\n')[0] if '\n' in actual_text else actual_text[:100],
                                                        "confidence": 0.7,
                                                        "explanation": actual_text,
                                                        "related_links": [],
                                                        "search_terms_used": search_terms_list
                                                    }
                                            except:
                                                # Fall back to using text directly
                                                context_data = {
                                                    "context_guess": actual_text.split('\n')[0] if '\n' in actual_text else actual_text[:100],
                                                    "confidence": 0.7,
                                                    "explanation": actual_text,
                                                    "related_links": [],
                                                    "search_terms_used": search_terms_list
                                                }
                                        else:
                                            # Just use the original parsed data
                                            context_data = parsed_data
                                else:
                                    context_data = parsed_data
                            else:
                                # Rest of existing code for non-JSON case
                                context_data = {
                                    "context_guess": state.context_inference.split('\n')[0] if '\n' in state.context_inference else state.context_inference[:100],
                                    "confidence": 0.7,
                                    "explanation": state.context_inference,
                                    "related_links": [],
                                    "search_terms_used": search_terms_list
                                }
                        except Exception as e:
                            state.log_step(f"DEBUG: Error extracting context: {str(e)}")
                            # Fallback data - VERY explicit to ensure we get something
                            context_data = {
                                "context_guess": "Unknown context",
                                "confidence": 0.5,
                                "explanation": str(state.context_inference or "No context inference available"),
                                "related_links": [],
                                "search_terms_used": search_terms_list
                            }
                            state.log_step("DEBUG: Using fallback context data")
                        
                        # Ensure all expected fields are present
                        if "context_guess" not in context_data:
                            context_data["context_guess"] = "Unspecified context"
                        if "confidence" not in context_data:
                            context_data["confidence"] = 0.5
                        if "explanation" not in context_data:
                            context_data["explanation"] = "No detailed explanation available."
                        if "related_links" not in context_data:
                            context_data["related_links"] = []
                        if "search_terms_used" not in context_data:
                            context_data["search_terms_used"] = search_terms_list
                            
                        # Ensure confidence is a float
                        try:
                            context_data["confidence"] = float(context_data["confidence"])
                        except (ValueError, TypeError):
                            context_data["confidence"] = 0.5
                            
                        # Store the structured output
                        state.final_output = context_data
                        state.log_step(f"✅ Final output formatting complete with guess: {context_data['context_guess']}")
                        state.log_step(f"DEBUG: Final output structure: {context_data.keys()}")
                    else:
                        state.log_step("⚠️ No context inference available for final output")
                        # Create a minimal fallback output
                        state.final_output = {
                            "context_guess": "Analysis incomplete",
                            "confidence": 0.3,
                            "explanation": "The analysis process didn't produce a complete context inference.",
                            "related_links": [],
                            "search_terms_used": []
                        }
                except Exception as e:
                    state.log_step(f"❌ Error formatting final output: {str(e)}")
                    # Create emergency fallback output
                    state.final_output = {
                        "context_guess": "Analysis error",
                        "confidence": 0.1,
                        "explanation": f"An error occurred during analysis: {str(e)}",
                        "related_links": [],
                        "search_terms_used": []
                    }

                # After all analysis is done
                state.log_step("Analysis complete.")
                progress_placeholder.text(state.get_progress_text())
                
                # Add this at the end, after final output is generated but before returning
                if state.final_output and image_hash:
                    state.log_step("💾 Storing analysis results in memory systems...")
                    try:
                        state.log_step("🧩 Preparing data for ChromaDB vector storage...")
                        store_result = await session.call_tool("store_analysis", 
                                                           arguments={"input_data": {
                                                               "image_hash": image_hash,
                                                               "analysis_json": json.dumps(state.final_output)
                                                           }})
                        
                        if hasattr(store_result, 'content') and store_result.content:
                            store_text = store_result.content[0].text
                            state.log_step("✅ Analysis vectorized and stored in ChromaDB")
                            
                            # Log collections that were updated
                            collections = []
                            if "visual_elements" in store_text:
                                collections.append("visual_elements")
                            if "style_analysis" in store_text:
                                collections.append("style_analysis")
                            if "scenario_analysis" in store_text:
                                collections.append("scenario_analysis")
                            if "complete_analysis" in store_text:
                                collections.append("complete_analysis")
                                
                            if collections:
                                state.log_step(f"📊 Vector embeddings created for: {', '.join(collections)}")
                                state.log_step(f"🗂️ Data indexed in ChromaDB for future similarity search")
                            else:
                                state.log_step("⚠️ No specific ChromaDB collections mentioned in storage response")
                        else:
                            state.log_step("⚠️ Failed to store analysis vectors in ChromaDB")
                    except Exception as e:
                        state.log_step(f"❌ ChromaDB storage error: {str(e)}")

                # Return the results
                return {
                    "raw_output": state.get_progress_text(),
                    "structured": state.final_output,
                    "error": state.errors[-1] if state.errors else None
                }

    except Exception as e:
        error_msg = f"Error during MCP session: {str(e)}"
        state.log_step(f"❌ Fatal Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        st.error(f"An error occurred during analysis: {e}")
        return {"raw_output": state.get_progress_text(), "structured": None, "error": error_msg}


def main():
    st.title("Context Detective")
    
    # Move sidebar code here, before image upload
    st.sidebar.title("User Preferences")
    interests_options = [
        "Art", "Technology", "Nature", "History", "Culture", "Science",
        "Current Events"
    ]
    user_interests = st.sidebar.multiselect(
        "Select your interests:",
        options=interests_options,
        default=["Art", "History"]
    )
    user_prefs = {"interests": user_interests}
    
    # Let user upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image")
        # Placeholder for progress
        progress_placeholder = st.empty()
        
        if st.button("Analyze"):
            import tempfile
            import asyncio

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # Run the async function in Streamlit
            result = asyncio.run(run_context_analysis(temp_path, user_prefs, progress_placeholder))
            
            st.subheader("Analysis Log")
            st.text_area("Log Output", result["raw_output"], height=200)

            if result["error"]:
                st.error(f"Error: {result['error']}")
            else:
                # Better display of structured output
                structured = result["structured"]
                
                if structured is None:
                    st.warning("Analysis completed but no structured output was produced.")
                else:
                    st.subheader("Context Analysis Results")
                    
                    # Main findings in a highlighted box
                    st.success(f"**Context:** {structured.get('context_guess', 'Unknown')}")
                    st.progress(float(structured.get('confidence', 0)))
                    st.write(f"Confidence: {int(float(structured.get('confidence', 0))*100)}%")
                    
                    # Explanation in a dedicated section
                    with st.expander("Detailed Explanation", expanded=True):
                        st.write(structured.get('explanation', 'No explanation available'))
                    
                    # Related links and search terms
                    if structured.get('related_links'):
                        with st.expander("Related Links"):
                            for link in structured.get('related_links', []):
                                st.markdown(f"- [{link}]({link})")
                    
                    # Show the raw JSON for reference
                    with st.expander("Raw JSON Result"):
                        st.json(structured)

if __name__ == "__main__":
    main()

# Function to load and prepare image for Gemini
def load_image_for_gemini(image_path):
    """Load and prepare an image for use with Gemini API."""
    try:
        img = Image.open(image_path)
        # For the genai client, we need an image object
        # Convert PIL Image to binary data
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        
        # Create image part for content
        image_part = {"mime_type": "image/png", "data": image_bytes}
        return image_part
    except Exception as e:
        logging.error(f"Error loading image for Gemini: {str(e)}")
        raise e

# Helper function to convert an image to base64
def image_to_base64(image_path):
    """Convert an image to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        raise e

# Add this helper to track processing state in app.py
async def run_memory_enhanced_workflow(session, state, image_path):
    """Run the full memory-enhanced workflow step by step."""
    try:
        # Step 1: Compute image hash
        state.log_step("Computing image hash...")
        hash_result = await session.call_tool("compute_image_hash", arguments={"input_data": {"image_path": image_path}})
        
        if not hash_result or not hasattr(hash_result, 'content'):
            state.log_step("❌ Failed to compute image hash")
            return False
            
        image_hash = hash_result.content[0].text
        state.update('image_hash', image_hash)
        state.log_step(f"✅ Image hash: {image_hash[:8]}")
        
        # Step 2: Check for exact match
        state.log_step("Checking memory for exact match...")
        match_result = await session.call_tool("check_exact_match", arguments={"input_data": {"image_hash": image_hash}})
        
        match_content = match_result.content[0].text if hasattr(match_result, 'content') else ""
        if "match_found" in match_content and "true" in match_content.lower():
            state.log_step("🎯 EXACT MATCH FOUND! Retrieving cached analysis...")
            # Parse the cached analysis
            try:
                cached_analysis = json.loads(match_content)
                state.update('final_output', cached_analysis)
                state.log_step("✅ Successfully retrieved cached analysis")
                return "cached"
            except:
                state.log_step("⚠️ Failed to parse cached analysis, continuing with new analysis")
        else:
            state.log_step("🔍 No exact match found, proceeding with analysis")
        
        # Continue with analysis as normal...
        # After each significant step, store in short-term memory

        # For visual elements
        if state.visual_elements:
            session_id = state.session_id
            store_in_short_term_memory(session_id, "visual_elements", state.visual_elements)
            
        # For style analysis
        if state.style_analysis:
            session_id = state.session_id
            store_in_short_term_memory(session_id, "style_analysis", state.style_analysis)
    
        # For scenario analysis
        if state.scenario_analysis:
            session_id = state.session_id
            store_in_short_term_memory(session_id, "scenario_analysis", state.scenario_analysis)
            
        # Before infer_context, call retrieve_memory_for_inference
        state.log_step("Retrieving similar analyses from memory to enhance inference...")
        memory_data = {
            "visual_elements": state.visual_elements,
            "style_analysis": state.style_analysis,
            "scenario_analysis": state.scenario_analysis
        }
        memory_result = await session.call_tool("retrieve_memory_for_inference", arguments={"input_data": memory_data})
        
        # ... complete workflow including infer_context

        # Finally, store the complete analysis
        if state.final_output:
            state.log_step("Storing final analysis in long-term memory...")
            store_result = await session.call_tool("store_analysis", arguments={
                "input_data": {
                    "image_hash": image_hash,
                    "analysis_json": json.dumps(state.final_output)
                }
            })
            state.log_step("✅ Analysis stored successfully in memory for future reuse")
            
        return True
    except Exception as e:
        state.log_step(f"❌ Error in memory-enhanced workflow: {str(e)}")
        return False
        
# Helper function for short-term memory
def store_in_short_term_memory(session_id, component, data):
    """Store analysis component in short-term memory via global memory module."""
    try:
        from main import memory
        memory.store_in_short_term(session_id, component, data)
        logger.info(f"✅ Stored {component} in short-term memory for session {session_id[:8]}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to store in short-term memory: {str(e)}")
        return False

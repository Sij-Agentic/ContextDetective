import streamlit as st
import asyncio
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
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
import re

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
    
    # Prepare the prompt
    # Prepare status notes based on the analysis state
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
    
    # Add a brief format reminder to the end of the prompt
    format_reminder = """
IMPORTANT FINAL REMINDER:
Your entire response MUST ONLY be in ONE of these EXACT formats:
1. FUNCTION_CALL: tool_name|parameter  (NO JSON, NO parameter names, NO explanations)
2. FINAL_ANSWER: {"json": "output"}

⚠️ CRITICAL WARNING: ⚠️
DO NOT include ANY explanation text before, during, or after your response.
NEVER say things like:
- "The image has been processed" 
- "Therefore no further action is required" 
- "Next, I will search the web"
- "Based on the image analysis"
- "I'll now call"
JUST THE RAW FUNCTION CALL OR FINAL ANSWER ONLY.

FOLLOW THE STRICT TOOL SEQUENCE:
describe_visual_elements -> describe_style_or_aesthetics -> describe_possible_scenario -> 
generate_search_terms -> search_web -> infer_context -> FINAL_ANSWER

BAD examples (DO NOT DO THESE):
- "I will analyze the image. FUNCTION_CALL: describe_visual_elements|path"
- "The image has been processed. Therefore, FUNCTION_CALL: search_web|query"
- "FUNCTION_CALL: visual_reasoning|{"image_path":"path"}"
- "FUNCTION_CALL: get_system_prompt"
- "FUNCTION_CALL: describe_style_or_aesthetics|image_path:path"
- "FUNCTION_CALL: generate_search_terms" (if you haven't called all three describe tools first)
- "I can now proceed to infer context. FUNCTION_CALL: infer_context"
- "Now that the search is complete, I will provide the final answer."

GOOD examples (DO EXACTLY LIKE THESE):
- "FUNCTION_CALL: describe_visual_elements|C:\\path\\to\\image.png"
- "FUNCTION_CALL: describe_style_or_aesthetics|C:\\path\\to\\image.png"
- "FUNCTION_CALL: describe_possible_scenario|C:\\path\\to\\image.png"
- "FUNCTION_CALL: generate_search_terms"
- "FUNCTION_CALL: search_web|flowers in japanese art history"
- "FUNCTION_CALL: infer_context"
- "FINAL_ANSWER: {\"context\":\"Japanese Ukiyo-e art from Edo period\",\"confidence\":0.85,\"explanation\":\"The visual elements and style match Ukiyo-e woodblock prints\"}"

# Add this to format_reminder in generate_agent_step function
The FINAL_ANSWER must be a JSON with this exact structure:
{
  "context_guess": "Brief description of the historical/cultural context",
  "confidence": 0.75,
  "explanation": "Detailed explanation of why this context is likely",
  "related_links": ["link1", "link2"],
  "search_terms_used": ["term1", "term2"]
}
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
            # Use simplified representation if schema is complex
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
    
    # Define workflow sequence - tools must be called in this order
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
    
    # Check if we're trying to call a tool that's already been successfully executed
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
    
    # Validate sequence - check that we're not skipping steps
    if tool_name in workflow_sequence:
        current_index = workflow_sequence.index(tool_name)
        
        # For each previous step in the sequence, check if it's been completed
        for i in range(current_index):
            previous_tool = workflow_sequence[i]
            
            # Check if the previous tool has been completed by checking the state
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
        # Normalize parameters - handle possible formatting issues
        normalized_params = []
        for p in params:
            # Special case for image paths with parameter names
            if "image_path" in p and ("C:" in p or "/" in p or "\\" in p):
                # Try to extract the path
                if "image_path=" in p:
                    path_value = p.split("image_path=", 1)[1].strip()
                    normalized_params.append(path_value)
                    state.log_step(f"Extracted path from image_path= format: {path_value}")
                elif "image_path:" in p:
                    path_value = p.split("image_path:", 1)[1].strip()
                    normalized_params.append(path_value)
                    state.log_step(f"Extracted path from image_path: format: {path_value}")
                else:
                    # If we can't extract clearly, just use the parameter
                    normalized_params.append(p.strip())
            # Handle other param=value cases
            elif "=" in p and not p.startswith("http"):
                normalized_params.append(p.split("=", 1)[1].strip())
            elif ":" in p and not (p.startswith("http") or p.startswith("C:\\") or p.startswith("c:\\")):
                # This is likely a param:value format, not a Windows path
                normalized_params.append(p.split(":", 1)[1].strip())
            else:
                # Clean up any quotes or extra formatting
                param = p.strip()
                if param.startswith('"') and param.endswith('"'):
                    param = param[1:-1]  # Remove surrounding quotes
                normalized_params.append(param)
        
        state.log_step(f"Normalized parameters: {normalized_params}")
        
        # Prepare input model based on tool name
        if tool_name == "describe_visual_elements" or tool_name == "describe_style_or_aesthetics" or tool_name == "describe_possible_scenario":
            # Perception tools take an image path
            if normalized_params:
                # Use last parameter as image path to handle different formatting patterns
                input_model = PerceptionInput(image_path=normalized_params[-1])
            else:
                state.log_step(f"❌ Error: No image path provided for {tool_name}")
                state.errors.append(f"TOOL ERROR: Tool {tool_name} requires an image path. Use the exact format: FUNCTION_CALL: {tool_name}|C:\\path\\to\\image.png")
                return f"Error: No image path provided for {tool_name}. Correct format: {tool_name}|image_path"
                
        elif tool_name == "generate_search_terms":
            # Collect descriptions from state
            descriptions = []
            if state.visual_elements:
                descriptions.append(state.visual_elements)
            if state.style_analysis:
                descriptions.append(state.style_analysis)  
            if state.scenario_analysis:
                descriptions.append(state.scenario_analysis)
                
            if not descriptions:
                error_msg = "Cannot generate search terms yet - no analysis results available"
                state.log_step(f"❌ {error_msg}")
                state.errors.append(f"TOOL ERROR: {error_msg}. First complete the visual analysis steps.")
                return f"Error: {error_msg}. Complete some analysis tools first."
                
            input_model = SearchTermsInput(descriptions=descriptions)
                
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
            
            # Create input model with string values
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
        
        # Convert Pydantic model to a dictionary before passing to call_tool
        if hasattr(input_model, "dict"):
            # This is a Pydantic model - convert to dict and wrap in input_data
            arguments_dict = {"input_data": input_model.dict()}
            result = await session.call_tool(tool_name, arguments=arguments_dict)
        else:
            # This is already a dictionary or other object
            result = await session.call_tool(tool_name, arguments={"input_data": input_model})
            
        client_logger.info(f"Received result from tool '{tool_name}'")

        # Extract text content from result
        texts = []
        if hasattr(result, 'content'):
            for item in result.content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                else:
                    # Handle potential non-text content if necessary
                    texts.append(str(item)) 
        else:
            texts.append(str(result))  # Fallback

        result_str = "\n".join(texts)
        state.log_step(f"Tool '{tool_name}' executed.")
        client_logger.debug(f"Tool Result Text (truncated): {result_str[:200]}...")
        
        # Add to run_context_analysis function after each tool call
        if result:
            # Safely handle result object without assuming it's a string
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
    
    state = WorkflowState()
    state.log_step("Starting context analysis...")
    progress_placeholder.text(state.get_progress_text())
    
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
    
    final_result_json = None

    try:
        # Connect to SSE server
        async with sse_client("http://127.0.0.1:8000/sse") as (reader, writer):
            state.log_step("SSE Connection established.")
            progress_placeholder.text(state.get_progress_text())
            
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                state.log_step("MCP Session initialized.")
                progress_placeholder.text(state.get_progress_text())

                # Get available tools
                tools_result = await session.list_tools()
                all_tools = tools_result.tools
                
                # Filter out system tools like get_system_prompt
                tools = [tool for tool in all_tools if tool.name != "get_system_prompt"]
                tools_block = format_tools_for_prompt(tools)
                
                # Try to get system prompt from server
                try:
                    system_prompt = await session.call_tool("get_system_prompt", {})
                    system_prompt = system_prompt.content[0].text if hasattr(system_prompt, 'content') else str(system_prompt)
                    state.log_step("Retrieved system prompt from server.")
                except Exception as e:
                    # Use the actual SYSTEM_PROMPT rather than a placeholder
                    system_prompt = """
FOLLOW THESE EXACT INSTRUCTIONS:

1. You must ONLY respond with ONE of these formats:
   FUNCTION_CALL: tool_name|parameter
   FINAL_ANSWER: {"json": "output"}

2. USE THE TOOLS IN THIS EXACT SEQUENCE - DO NOT SKIP STEPS:
   1️⃣ FIRST: FUNCTION_CALL: describe_visual_elements|C:\\path\\to\\image.png
   2️⃣ SECOND: FUNCTION_CALL: describe_style_or_aesthetics|C:\\path\\to\\image.png
   3️⃣ THIRD: FUNCTION_CALL: describe_possible_scenario|C:\\path\\to\\image.png
   4️⃣ FOURTH: FUNCTION_CALL: generate_search_terms
   5️⃣ FIFTH: FUNCTION_CALL: search_web|search query
   6️⃣ SIXTH: FUNCTION_CALL: infer_context
   7️⃣ LAST: FINAL_ANSWER: {"json output"}

CRITICAL: FOLLOW THE EXACT SEQUENCE ABOVE. You MUST complete describe_visual_elements, describe_style_or_aesthetics, and describe_possible_scenario BEFORE calling generate_search_terms.

3. DO NOT add ANY explanations before or after your function call.
4. DO NOT use ANY other format or tools than those listed above.
5. DO NOT include "image_path" or any parameter names in your function calls.

Example: FUNCTION_CALL: describe_visual_elements|C:\\Users\\path\\to\\image.png
"""
                    system_prompt = system_prompt.replace('{tools_block}', tools_block)
                    state.log_step("Using default system prompt.")
                
                # Main workflow loop
                max_iterations = 10  # Increase max iterations
                mandatory_tools_completed = {
                    "describe_visual_elements": False,
                    "describe_style_or_aesthetics": False,
                    "describe_possible_scenario": False,
                    "generate_search_terms": False,
                    "search_web": False,
                    "infer_context": False
                }
                
                # Track repeated tool calls to detect loops
                tool_call_history = []
                repeated_call_threshold = 2  # Number of consecutive repeated calls to consider as stuck
                
                while state.iteration < max_iterations:
                    state.iteration += 1
                    state.log_step(f"--- Starting Agent Iteration {state.iteration} ---")
                    progress_placeholder.text(state.get_progress_text())

                    # Track if we're in the final iterations and need to ensure all tools are called
                    approaching_max = state.iteration >= max_iterations - 3
                    if approaching_max:
                        missing_tools = [tool for tool, completed in mandatory_tools_completed.items() if not completed]
                        if missing_tools:
                            state.log_step(f"Approaching max iterations. Still need to complete: {', '.join(missing_tools)}")
                            
                    # Update completed tools based on the current state
                    if state.visual_elements:
                        mandatory_tools_completed["describe_visual_elements"] = True
                    if state.style_analysis:
                        mandatory_tools_completed["describe_style_or_aesthetics"] = True
                    if state.scenario_analysis:
                        mandatory_tools_completed["describe_possible_scenario"] = True
                    if state.search_terms:
                        mandatory_tools_completed["generate_search_terms"] = True
                    if state.web_findings:
                        mandatory_tools_completed["search_web"] = True
                    if state.context_inference:
                        mandatory_tools_completed["infer_context"] = True
                    
                    # Construct prompt for the agent
                    prompt = f"{system_prompt}\n\nImage Path: {image_path}\n"
                    if state.errors:
                        prompt += "\nPrevious Errors:\n" + "\n".join(state.errors)
                        state.errors = []  # Clear errors after showing them
                    
                    if approaching_max and missing_tools:
                        prompt += f"\n\nIMPORTANT: You must call these mandatory tools to complete the analysis: {', '.join(missing_tools)}"
                        
                    prompt += "\nWhat is the next step based on the workflow?"

                    # Add state information to help the agent
                    if state.visual_elements:
                        prompt += f"\n\nCurrent Visual Elements: {state.visual_elements[:500]}..."
                    if state.style_analysis:
                        prompt += f"\n\nCurrent Style Analysis: {state.style_analysis[:500]}..."
                    if state.scenario_analysis:
                        prompt += f"\n\nCurrent Scenario Analysis: {state.scenario_analysis[:500]}..."
                    if state.search_terms:
                        prompt += f"\n\nCurrent Search Terms: {state.search_terms}"
                    if state.web_findings:
                        prompt += f"\n\nCurrent Web Findings: {state.web_findings[:500]}..."
                    if state.context_inference:
                        prompt += f"\n\nCurrent Context Inference: {state.context_inference[:500]}..."
                    
                    # Check for repeated tool calls and provide explicit guidance
                    is_stuck = False
                    if len(tool_call_history) >= repeated_call_threshold:
                        # Check if the last N calls were the same
                        last_calls = tool_call_history[-repeated_call_threshold:]
                        if all(call == last_calls[0] for call in last_calls):
                            is_stuck = True
                            state.log_step(f"⚠️ Detected repetition of tool: {last_calls[0]}. Providing explicit guidance.")
                            
                            # Find the next tool to recommend
                            next_tool = None
                            for tool in ["describe_visual_elements", "describe_style_or_aesthetics", "describe_possible_scenario", 
                                        "generate_search_terms", "search_web", "infer_context"]:
                                if not mandatory_tools_completed.get(tool, False):
                                    next_tool = tool
                                    break
                            
                            if next_tool:
                                prompt += f"\n\n⚠️ IMPORTANT: You are repeating the same tool call. You should move to the next step: {next_tool}"
                                if next_tool in ["describe_visual_elements", "describe_style_or_aesthetics", "describe_possible_scenario"]:
                                    prompt += f"\nUse: FUNCTION_CALL: {next_tool}|{image_path}"
                                else:
                                    prompt += f"\nUse: FUNCTION_CALL: {next_tool}"
                    
                    # If we have all mandatory tools and we're past iteration 5, suggest final answer
                    all_mandatory_complete = all(mandatory_tools_completed.values())
                    if all_mandatory_complete and state.iteration >= 5:
                        prompt += "\n\nAll mandatory tools have been called. You should now provide a FINAL_ANSWER."

                    client_logger.debug(f"Iteration {state.iteration} Prompt: {prompt}")

                    # Generate next step
                    try:
                        # Prepare the analysis state with all current data
                        analysis_state = {}
                        if state.visual_elements:
                            analysis_state["visual_elements"] = state.visual_elements
                        if state.style_analysis:
                            analysis_state["style_or_aesthetics"] = state.style_analysis
                        if state.scenario_analysis:
                            analysis_state["possible_scenario"] = state.scenario_analysis
                        if state.search_terms:
                            analysis_state["search_terms"] = state.search_terms
                        if state.web_findings:
                            analysis_state["search_results"] = state.web_findings
                        if state.context_inference:
                            analysis_state["inferred_context"] = state.context_inference
                            
                        # Prepare history for context
                        history = state.analysis_log.copy()
                        
                        # Call generate_agent_step with the required parameters
                        try:
                            agent_response = await asyncio.to_thread(
                                generate_agent_step,
                                session,
                                image_path,
                                analysis_state,
                                history
                            )
                            state.log_step("Agent response received.")
                        except Exception as e:
                            error_msg = f"Error generating agent step: {str(e)}"
                            state.log_step(f"❌ {error_msg}")
                            client_logger.error(error_msg, exc_info=True)
                            agent_response = f"FUNCTION_CALL: describe_visual_elements|{image_path}"
                        
                        # Debug log to see the actual response
                        client_logger.debug(f"Raw agent response: {agent_response}")
                        
                        # Log a sample of the response to help debug
                        first_200 = agent_response[:200] + ("..." if len(agent_response) > 200 else "")
                        state.log_step(f"Agent response sample: {first_200}")
                        
                        progress_placeholder.text(state.get_progress_text())
                        
                        # Extract function calls and final answers using regex patterns
                        function_match = re.search(r'FUNCTION_CALL:\s*([a-z_]+)\|?(.*?)(?:\s*$|\n)', agent_response, re.DOTALL)
                        final_answer_match = re.search(r'FINAL_ANSWER:\s*({.*})', agent_response, re.DOTALL)
                        
                        # Check for final answer
                        if final_answer_match:
                            answer_part = final_answer_match.group(1).strip()
                            state.log_step(f"Agent provided FINAL_ANSWER.")
                            client_logger.info(f"Final Answer Raw: {answer_part}")
                            state.update('final_output', answer_part)
                            final_result_json = answer_part  # Store the final JSON
                            break  # Exit loop
                        
                        # Process function calls - more flexible parsing
                        function_calls = []
                        if function_match:
                            function_part = function_match.group(0).split("FUNCTION_CALL:", 1)[1].strip()
                            function_calls.append(function_part)
                        else:
                            # Try alternative pattern to find function calls
                            alt_patterns = [
                                r'(?:FUNCTION_CALL|function_call|Function_Call):\s*([a-z_]+)\|?(.*?)(?:\s*$|\n)',
                                r'(?:Use|Call|Execute)\s+([a-z_]+)\s+with\s+(.*?)(?:\s*$|\n)',
                                r'([a-z_]+)\|([^|\n]+)(?:\s*$|\n)'
                            ]
                            
                            for pattern in alt_patterns:
                                alt_matches = re.findall(pattern, agent_response, re.IGNORECASE | re.DOTALL)
                                if alt_matches:
                                    for match in alt_matches:
                                        tool_name = match[0].strip()
                                        tool_input = match[1].strip() if len(match) > 1 else ""
                                        
                                        # Validate tool name
                                        if tool_name in ["describe_visual_elements", "describe_style_or_aesthetics", 
                                                        "describe_possible_scenario", "generate_search_terms", 
                                                        "search_web", "infer_context", "format_final_output"]:
                                            function_calls.append(f"{tool_name}|{tool_input}")
                                            state.log_step(f"Found function call using alternative pattern: {tool_name}")
                                            break

                        if not function_calls:
                            state.log_step("Agent did not provide FUNCTION_CALL or FINAL_ANSWER. Retrying with clearer instructions.")
                            
                            # Define workflow tools in order
                            workflow_tools = [
                                "describe_visual_elements", 
                                "describe_style_or_aesthetics",
                                "describe_possible_scenario",
                                "generate_search_terms",
                                "search_web",
                                "infer_context",
                                "format_final_output"
                            ]
                            
                            # Check which step we're at
                            next_tool = None
                            if not state.visual_elements:
                                next_tool = workflow_tools[0]
                            elif not state.style_analysis:
                                next_tool = workflow_tools[1]
                            elif not state.scenario_analysis:
                                next_tool = workflow_tools[2]
                            elif not state.search_terms:
                                next_tool = workflow_tools[3]
                            elif not state.web_findings:
                                next_tool = workflow_tools[4]
                            elif not state.context_inference:
                                next_tool = workflow_tools[5]
                            else:
                                next_tool = workflow_tools[6]
                                
                            # Create a specific suggestion
                            suggestion = f"You must call '{next_tool}' now. "
                            if next_tool == "describe_visual_elements" or next_tool == "describe_style_or_aesthetics" or next_tool == "describe_possible_scenario":
                                suggestion += f"Example: FUNCTION_CALL: {next_tool}|{state.image_path}"
                            elif next_tool == "generate_search_terms" or next_tool == "infer_context":
                                suggestion += f"Example: FUNCTION_CALL: {next_tool}"
                            elif next_tool == "search_web":
                                suggestion += f"Example: FUNCTION_CALL: {next_tool}|your search query"
                            else:
                                suggestion += "Use the format: FUNCTION_CALL: tool_name|parameter"
                            
                            # Add previous response analysis if available
                            response_analysis = ""
                            if "visual_reasoning" in agent_response:
                                response_analysis = "Your response used 'visual_reasoning' which is not an allowed tool. "
                            elif "get_system_prompt" in agent_response:
                                response_analysis = "Your response tried to call 'get_system_prompt' which is not an allowed tool. "
                            elif "image_path:" in agent_response or "image_path=" in agent_response:
                                response_analysis = "Your response included 'image_path:' or 'image_path=' which is incorrect format. "
                            elif "{" in agent_response and "}" in agent_response:
                                response_analysis = "Your response used JSON format which is incorrect. Do not use JSON for function calls. "
                            
                            state.errors.append(f"FORMAT ERROR: Your response must be EXACTLY in this format:\n" +
                                             f"FUNCTION_CALL: tool_name|parameter\n" +
                                             f"Do not include ANY other text. {response_analysis}{suggestion}")
                            # Don't break, retry with clearer instructions
                            continue
                        
                        # Execute calls
                        for func_call in function_calls:
                            state.log_step(f"Processing: {func_call}")
                            progress_placeholder.text(state.get_progress_text())
                            
                            # More flexible parsing - handles with or without pipe separator
                            parts = []
                            if "|" in func_call:
                                parts = [p.strip() for p in func_call.split("|") if p.strip()]
                            else:
                                # Try space separator as fallback
                                parts = [p.strip() for p in func_call.split() if p.strip()]
                                
                            if not parts:
                                state.log_step(f"Warning: Could not parse function call: {func_call}")
                                continue
                                
                            func_name = parts[0]
                            params = parts[1:] if len(parts) > 1 else []
                            
                            # Handle special case for tools that don't need parameters
                            if func_name == "generate_search_terms" and not params:
                                # No need to add dummy parameters - the execute_tool_call handles this
                                pass
                            elif func_name == "infer_context" and not params:
                                # No need to add dummy parameters - the execute_tool_call handles this
                                pass
                            
                            # Execute tool
                            tool_result = await execute_tool_call(session, func_name, params, state)
                            
                            # Update tool call history to track repetition
                            tool_call_history.append(func_name)
                            
                            # Update state based on tool result
                            if tool_result:
                                state.log_step(f"Tool '{func_name}' result processed.")
                                # Update state based on tool name
                                if func_name == "describe_visual_elements": 
                                    state.update('visual_elements', tool_result)
                                elif func_name == "describe_style_or_aesthetics": 
                                    state.update('style_analysis', tool_result)
                                elif func_name == "describe_possible_scenario": 
                                    state.update('scenario_analysis', tool_result)
                                elif func_name == "generate_search_terms": 
                                    state.update('search_terms', tool_result)
                                elif func_name == "search_web": 
                                    state.update('web_findings', tool_result)
                                elif func_name == "infer_context": 
                                    state.update('context_inference', tool_result)
                                elif func_name == "format_final_output": 
                                    state.update('final_output', tool_result)
                            else:
                                state.log_step(f"Tool '{func_name}' execution failed or returned no result.")

                            progress_placeholder.text(state.get_progress_text())
                    except Exception as e:
                        error_msg = f"Error during agent iteration {state.iteration}: {str(e)}"
                        state.log_step(f"❌ {error_msg}")
                        client_logger.error(error_msg, exc_info=True)
                        state.errors.append(error_msg)
                
                # After the loop ends
                if state.iteration >= max_iterations:
                    state.log_step("Reached maximum iterations. Stopping.")
                
                # Check if we completed all necessary steps
                missing_tools = [tool for tool, completed in mandatory_tools_completed.items() if not completed]
                if missing_tools:
                    state.log_step(f"Warning: Analysis incomplete. Missing tools: {', '.join(missing_tools)}")
                    
                if not final_result_json:
                    state.log_step("Workflow finished without a FINAL_ANSWER. Generating one based on available data.")
                    
                    # If we have enough data, try to generate a final output ourselves
                    if state.context_inference:
                        # We have context inference, so we can create a reasonable output
                        try:
                            context_parts = state.context_inference.split("\n")
                            context_guess = context_parts[0] if len(context_parts) > 0 else "Analysis incomplete"
                            explanation = state.context_inference
                            search_terms = state.search_terms.split("\n") if state.search_terms else []
                            
                            final_output = {
                                "context_guess": context_guess[:100],  # First line or part of first line
                                "confidence": 0.5,  # Medium confidence since this is auto-generated
                                "explanation": explanation,
                                "related_links": [],  # No links available if we didn't complete
                                "search_terms_used": search_terms
                            }
                            
                            final_result_json = json.dumps(final_output)
                            state.log_step("Auto-generated final output based on incomplete analysis.")
                        except Exception as auto_gen_error:
                            state.log_step(f"Failed to auto-generate output: {str(auto_gen_error)}")
                            final_result_json = json.dumps({
                                "context_guess": "Analysis incomplete",
                                "confidence": 0.1,
                                "explanation": "Workflow did not complete successfully.",
                                "related_links": [],
                                "search_terms_used": []
                            })
                    else:
                        final_result_json = json.dumps({
                            "context_guess": "Analysis incomplete",
                            "confidence": 0.1,
                            "explanation": "Workflow did not complete successfully.",
                            "related_links": [],
                            "search_terms_used": []
                        })

    except Exception as e:
        error_msg = f"Error during MCP session: {str(e)}"
        state.log_step(f"❌ Fatal Error: {error_msg}")
        logger.error(error_msg, exc_info=True)
        st.error(f"An error occurred during analysis: {e}")
        return {"raw_output": state.get_progress_text(), "structured": None, "error": error_msg}

    state.log_step("Analysis complete.")
    progress_placeholder.text(state.get_progress_text())

    # Parse the final JSON result
    structured_result = None
    error_parsing = None
    try:
        if final_result_json:
            # First parse the outer JSON
            structured_data = json.loads(final_result_json)
            
            # Check if we have a nested JSON in 'json' field
            if isinstance(structured_data, dict) and 'json' in structured_data:
                # Parse the inner stringified JSON
                inner_json = json.loads(structured_data['json'])
                structured_data = inner_json
            
            try:
                structured_result = ActionOutput(**structured_data).dict()
                state.log_step("Final JSON parsed and validated successfully.")
            except Exception as pydantic_error:
                state.log_step(f"Warning: Final JSON parsed but failed Pydantic validation: {pydantic_error}")
                structured_result = structured_data
        else:
            state.log_step("No final JSON output received from the agent.")
            
    except json.JSONDecodeError as json_err:
        state.log_step(f"❌ Error parsing final JSON output: {json_err}")
        error_parsing = str(json_err)
        client_logger.error(f"Failed to parse FINAL_ANSWER JSON: {final_result_json}", exc_info=True)
    except Exception as e:
        state.log_step(f"❌ Error processing final output: {e}")
        error_parsing = str(e)
        client_logger.error(f"Unexpected error processing final output: {final_result_json}", exc_info=True)

    return {
        "raw_output": state.get_progress_text(),
        "structured": structured_result,
        "error": state.errors[-1] if state.errors else error_parsing
    }


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
                st.subheader("Context Analysis Results")
                
                # Better display of structured output
                structured = result["structured"]
                
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

 import logging
import datetime
import os
import re
import json
import traceback
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamp for the log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"debug_logger_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG level for maximum information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger("DebugLogger")
logger.setLevel(logging.DEBUG)

# Log the log file path
logger.info(f"Debug logger initialized. Log file: {log_file}")

def debug_log_variable(name, value, max_length=1000):
    """Log a variable's content with truncation for large values"""
    if isinstance(value, str):
        if len(value) > max_length:
            logger.debug(f"{name} (truncated): {value[:max_length]}... (length: {len(value)})")
            # Also log the last part of the string
            logger.debug(f"{name} (end): ...{value[-max_length:]}")
        else:
            logger.debug(f"{name}: {value}")
    else:
        logger.debug(f"{name}: {value}")

def save_raw_output(output, prefix="raw_output"):
    """Save raw output to a file for inspection"""
    output_file = log_dir / f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)
    logger.info(f"Saved raw output to {output_file}")
    return output_file

def analyze_output_structure(output):
    """Analyze the structure of the output to identify patterns"""
    logger.info("Analyzing output structure")
    
    # Save the raw output
    output_file = save_raw_output(output, "structure_analysis")
    
    # Check for common patterns
    patterns = {
        "context_inference": r'context_inference:.*?text":\s*"(.*?)",\s*"annotations"',
        "confidence_score": r'Confidence Score:\s*(\d+\.\d+)',
        "context_guess": r'Most Likely Context:.*?\n\n(.*?)(?:\n\n|\Z)',
        "final_output": r'Final Output:(.*?)(?:\n\n|\Z)',
        "json_structure": r'\[{"content":\s*\[{"type":\s*"text",\s*"text":\s*"(.*?)",\s*"annotations"',
        "visual_elements": r'Visual Elements Analysis:',
        "style_analysis": r'Style Analysis:',
        "scenario_analysis": r'Scenario Analysis:',
    }
    
    # Check each pattern
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            logger.info(f"Found {len(matches)} matches for pattern '{pattern_name}'")
            if len(matches) > 0:
                # Log the first match (truncated if too long)
                match_preview = str(matches[0])
                if len(match_preview) > 200:
                    match_preview = match_preview[:200] + "..."
                logger.info(f"First match for '{pattern_name}': {match_preview}")
        else:
            logger.info(f"No matches found for pattern '{pattern_name}'")
    
    # Check for JSON structure
    try:
        # Try to find JSON-like structures
        json_matches = re.findall(r'\{.*?\}', output, re.DOTALL)
        if json_matches:
            logger.info(f"Found {len(json_matches)} potential JSON objects")
            for i, json_str in enumerate(json_matches[:3]):  # Log only the first 3
                try:
                    json_obj = json.loads(json_str)
                    logger.info(f"JSON object {i+1} is valid: {json.dumps(json_obj, indent=2)[:200]}...")
                except json.JSONDecodeError:
                    logger.info(f"JSON object {i+1} is not valid JSON")
        else:
            logger.info("No JSON-like structures found")
    except Exception as e:
        logger.error(f"Error analyzing JSON structure: {str(e)}")
    
    # Check for line breaks and formatting
    line_count = output.count('\n')
    logger.info(f"Output contains {line_count} line breaks")
    
    # Check for common delimiters
    delimiters = ['\n\n', '---', '===', '***']
    for delimiter in delimiters:
        count = output.count(delimiter)
        logger.info(f"Found {count} occurrences of delimiter '{delimiter}'")
    
    # Check for common section headers
    section_headers = ['Analysis:', 'Results:', 'Conclusion:', 'Summary:']
    for header in section_headers:
        count = output.count(header)
        logger.info(f"Found {count} occurrences of section header '{header}'")
    
    return output_file

def extract_context_from_logs(log_text):
    """Extract context information from the log output with detailed logging"""
    logger.info("Starting context extraction from logs")
    debug_log_variable("log_text_length", len(log_text))
    
    # Save the raw log text
    raw_log_file = save_raw_output(log_text, "raw_log")
    
    # Look for context inference section
    logger.info("Searching for context inference section")
    context_match = re.search(r'context_inference:.*?text":\s*"(.*?)",\s*"annotations"', log_text, re.DOTALL)
    
    if context_match:
        logger.info("Found context inference section")
        context_text = context_match.group(1)
        # Unescape the text
        context_text = context_text.replace('\\n', '\n').replace('\\"', '"')
        debug_log_variable("context_text", context_text)
        
        # Extract confidence score
        logger.info("Extracting confidence score")
        confidence_match = re.search(r'Confidence Score:\s*(\d+\.\d+)', context_text)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            logger.info(f"Extracted confidence score: {confidence}")
        else:
            confidence = 0.5
            logger.warning("Could not extract confidence score, using default value: 0.5")
        
        # Extract context guess
        logger.info("Extracting context guess")
        context_guess_match = re.search(r'Most Likely Context:.*?\n\n(.*?)(?:\n\n|\Z)', context_text, re.DOTALL)
        if context_guess_match:
            context_guess = context_guess_match.group(1).strip()
            logger.info(f"Extracted context guess: {context_guess[:100]}...")
        else:
            context_guess = "Context analysis completed"
            logger.warning("Could not extract context guess, using default value")
        
        # Create a result object
        result = {
            "context_guess": context_guess,
            "confidence": confidence,
            "explanation": context_text,
            "related_links": [],
            "search_terms_used": []
        }
        
        logger.info("Successfully created result object")
        return result
    
    # If we couldn't find the context inference section, try alternative patterns
    logger.warning("Could not find context inference section with primary pattern")
    
    # Try alternative pattern 1: Look for "Most Likely Context" directly
    logger.info("Trying alternative pattern 1: 'Most Likely Context'")
    alt_match1 = re.search(r'Most Likely Context:.*?\n\n(.*?)(?:\n\n|\Z)', log_text, re.DOTALL)
    if alt_match1:
        logger.info("Found 'Most Likely Context' with alternative pattern 1")
        context_guess = alt_match1.group(1).strip()
        logger.info(f"Extracted context guess: {context_guess[:100]}...")
        
        # Try to find confidence score
        confidence_match = re.search(r'Confidence Score:\s*(\d+\.\d+)', log_text)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            logger.info(f"Extracted confidence score: {confidence}")
        else:
            confidence = 0.5
            logger.warning("Could not extract confidence score, using default value: 0.5")
        
        # Create a result object
        result = {
            "context_guess": context_guess,
            "confidence": confidence,
            "explanation": log_text[:1000] + "...",  # Use the first 1000 chars as explanation
            "related_links": [],
            "search_terms_used": []
        }
        
        logger.info("Successfully created result object with alternative pattern 1")
        return result
    
    # Try alternative pattern 2: Look for JSON-like structure
    logger.info("Trying alternative pattern 2: JSON-like structure")
    json_match = re.search(r'\[{"content":\s*\[{"type":\s*"text",\s*"text":\s*"(.*?)",\s*"annotations"', log_text, re.DOTALL)
    if json_match:
        logger.info("Found JSON-like structure with alternative pattern 2")
        json_text = json_match.group(1)
        # Unescape the text
        json_text = json_text.replace('\\n', '\n').replace('\\"', '"')
        debug_log_variable("json_text", json_text)
        
        # Try to find confidence score
        confidence_match = re.search(r'Confidence Score:\s*(\d+\.\d+)', json_text)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            logger.info(f"Extracted confidence score: {confidence}")
        else:
            confidence = 0.5
            logger.warning("Could not extract confidence score, using default value: 0.5")
        
        # Try to find context guess
        context_guess_match = re.search(r'Most Likely Context:.*?\n\n(.*?)(?:\n\n|\Z)', json_text, re.DOTALL)
        if context_guess_match:
            context_guess = context_guess_match.group(1).strip()
            logger.info(f"Extracted context guess: {context_guess[:100]}...")
        else:
            context_guess = "Context analysis completed"
            logger.warning("Could not extract context guess, using default value")
        
        # Create a result object
        result = {
            "context_guess": context_guess,
            "confidence": confidence,
            "explanation": json_text,
            "related_links": [],
            "search_terms_used": []
        }
        
        logger.info("Successfully created result object with alternative pattern 2")
        return result
    
    # If all patterns fail, log the failure and return None
    logger.error("All extraction patterns failed. Could not extract context information from logs.")
    logger.error("Log text sample (first 500 chars): " + log_text[:500])
    logger.error("Log text sample (last 500 chars): " + log_text[-500:])
    
    # Save the log text to a file for further analysis
    failed_log_file = save_raw_output(log_text, "failed_extraction")
    
    return None

def log_process_output(process_output, process_name="client"):
    """Log the output from a process with detailed information"""
    logger.info(f"Logging output from {process_name} process")
    
    # Save the raw output
    output_file = save_raw_output(process_output, f"{process_name}_output")
    
    # Log the output length
    logger.info(f"{process_name} output length: {len(process_output)}")
    
    # Log the first and last 500 characters
    logger.info(f"{process_name} output preview (first 500 chars): {process_output[:500]}")
    logger.info(f"{process_name} output preview (last 500 chars): {process_output[-500:]}")
    
    # Analyze the output structure
    structure_file = analyze_output_structure(process_output)
    
    # Try to extract context information
    result = extract_context_from_logs(process_output)
    
    if result:
        logger.info(f"Successfully extracted context information: {result['context_guess']}")
        return result
    
    logger.warning(f"Could not extract context information from {process_name} output")
    return None

# Example usage
if __name__ == "__main__":
    # This is just an example of how to use the debug logger
    logger.info("Debug logger is ready to use")
    
    # Example: Analyze a sample output
    sample_output = """
    Visual Elements Analysis:
    - Detected text: "Sample text"
    - Detected objects: car, person
    
    Style Analysis:
    - Color scheme: bright
    - Composition: centered
    
    Scenario Analysis:
    - Possible scenarios: outdoor, urban
    
    Context Inference:
    Most Likely Context: This is a sample context
    
    Confidence Score: 0.85
    
    Final Output:
    This is the final output of the analysis.
    """
    
    result = log_process_output(sample_output, "sample")
    
    if result:
        logger.info(f"Sample analysis result: {result}")
    else:
        logger.error("Failed to analyze sample output")
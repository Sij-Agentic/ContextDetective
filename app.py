import streamlit as st
import asyncio
import os
from pathlib import Path
import tempfile
from PIL import Image
import json
import subprocess
import sys
import time
from dotenv import load_dotenv
import logging
import datetime
import re
import threading
import queue
import traceback

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
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"streamlit_app_{timestamp}.log"
    
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
    logger = logging.getLogger("ContextDetectiveUI")
    logger.setLevel(logging.DEBUG)
    
    # Log the log file path
    logger.info(f"Log file created at: {log_file}")
    
    return logger

logger = setup_logging()

# Function to debug log the content of a variable
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

# Function to run the MCP server as a subprocess
def start_mcp_server():
    try:
        # Start the MCP server as a subprocess
        server_process = subprocess.Popen(
            [sys.executable, "example2-detective.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("MCP server started successfully")
        return server_process
    except Exception as e:
        logger.error(f"Failed to start MCP server: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Failed to start MCP server: {str(e)}")
        return None

# Function to read output from a pipe in a separate thread
def read_output(pipe, output_queue):
    for line in pipe:
        output_queue.put(line)
    pipe.close()

# Function to extract context information from the log output
def extract_context_from_logs(log_text):
    logger.info("Starting context extraction from logs")
    debug_log_variable("log_text_length", len(log_text))
    
    # Save the raw log text to a file for inspection
    raw_log_file = Path("logs") / f"raw_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(raw_log_file, "w", encoding="utf-8") as f:
        f.write(log_text)
    logger.info(f"Saved raw log text to {raw_log_file}")
    
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
    failed_log_file = Path("logs") / f"failed_extraction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(failed_log_file, "w", encoding="utf-8") as f:
        f.write(log_text)
    logger.info(f"Saved failed extraction log to {failed_log_file}")
    
    return None

# Function to run the client with the image path
def run_client_with_image(image_path, progress_placeholder):
    try:
        # Log the command being executed
        logger.info(f"Running client with image path: {image_path}")
        progress_placeholder.info("Starting image analysis...")
        
        # Run the client with the image path using subprocess
        process = subprocess.Popen(
            [sys.executable, "talk2mcp-detective.py", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Create queues for stdout and stderr
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        # Create threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue))
        
        # Start the threads
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Create a container for real-time logs
        log_container = progress_placeholder.empty()
        log_text = ""
        
        # Function to update the log display
        def update_log_display():
            nonlocal log_text
            log_container.text(log_text)
        
        # Read output in real-time
        start_time = time.time()
        last_update_time = start_time
        
        # Collect all output
        all_stdout = ""
        all_stderr = ""
        
        # Log the start of the process
        logger.info("Starting to read client process output")
        
        while process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty():
            # Check for stdout
            try:
                while not stdout_queue.empty():
                    line = stdout_queue.get_nowait()
                    all_stdout += line
                    log_text += line + "\n"
                    
                    # Log important lines
                    if "Visual Elements Analysis" in line or "Style Analysis" in line or "Scenario Analysis" in line or "Context Inference" in line or "Final Output" in line:
                        logger.info(f"Important line detected: {line.strip()}")
                    
                    # Check for specific progress indicators
                    if "Visual Elements Analysis" in line:
                        progress_placeholder.info("Analyzing visual elements...")
                    elif "Style Analysis" in line:
                        progress_placeholder.info("Analyzing style and aesthetics...")
                    elif "Scenario Analysis" in line:
                        progress_placeholder.info("Analyzing possible scenarios...")
                    elif "Context Inference" in line:
                        progress_placeholder.info("Inferring context...")
                    elif "Final Output" in line:
                        progress_placeholder.info("Formatting final output...")
            except queue.Empty:
                pass
            
            # Check for stderr
            try:
                while not stderr_queue.empty():
                    line = stderr_queue.get_nowait()
                    all_stderr += line
                    log_text += f"ERROR: {line}\n"
                    logger.error(f"Client stderr: {line.strip()}")
            except queue.Empty:
                pass
            
            # Update the display every 0.5 seconds
            current_time = time.time()
            if current_time - last_update_time > 0.5:
                update_log_display()
                last_update_time = current_time
            
            # Add a small delay to prevent CPU hogging
            time.sleep(0.1)
        
        # Wait for the process to complete
        process.wait()
        
        # Log the return code
        logger.info(f"Client process return code: {process.returncode}")
        
        if process.returncode != 0:
            logger.error(f"Client process failed with return code {process.returncode}: {all_stderr}")
            progress_placeholder.error(f"Analysis failed: {all_stderr}")
            return None
        
        # Save the raw output to a file for inspection
        output_file = Path("logs") / f"client_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(log_text)  # Use log_text instead of all_stdout
        logger.info(f"Saved raw output to {output_file}")
        
        # Extract the last line of output from log_text
        lines = log_text.strip().split('\n')
        if not lines:
            logger.error("No output lines found")
            return {
                "raw_output": log_text,
                "stderr": all_stderr,
                "structured": None
            }
        
        last_line = lines[-1]
        logger.info(f"Last line of output: {last_line[:100]}...")
        
        # Use a language model to structure the output
        structured_result = structure_output_with_language_model(last_line)
        
        return {
            "raw_output": log_text,
            "stderr": all_stderr,
            "structured": structured_result
        }
            
    except Exception as e:
        logger.error(f"Error running client: {str(e)}", exc_info=True)
        progress_placeholder.error(f"Error during analysis: {str(e)}")
        return None

def structure_output_with_language_model(output_text):
    """
    Use a language model to structure the output text into a standardized format.
    This is a placeholder function that should be replaced with an actual API call.
    """
    try:
        # For now, we'll use a simple regex-based approach
        # In a real implementation, this would call an API like OpenAI's GPT
        
        # Extract confidence score
        confidence_match = re.search(r'Confidence Score:\s*(\d+\.\d+)', output_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Extract context guess
        context_guess_match = re.search(r'Most Likely Context:.*?\n\n(.*?)(?:\n\n|\Z)', output_text, re.DOTALL)
        context_guess = context_guess_match.group(1).strip() if context_guess_match else "Context analysis completed"
        
        # Create structured result
        structured_result = {
            "context_guess": context_guess,
            "confidence": confidence,
            "explanation": output_text,
            "related_links": [],
            "search_terms_used": []
        }
        
        logger.info("Successfully structured output with language model")
        return structured_result
    except Exception as e:
        logger.error(f"Error structuring output with language model: {str(e)}")
        return None

def main():
    st.title("Context Detective")
    st.write("Upload an image to analyze its context.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Create a placeholder for progress updates
        progress_placeholder = st.empty()
        
        # Run the analysis
        result = run_client_with_image(tmp_file_path, progress_placeholder)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        if result:
            # Display the raw output in a code block
            st.subheader("Raw Output")
            st.code(result["raw_output"], language="text")
            
            # If there was any stderr, display it
            if result["stderr"]:
                st.subheader("Errors/Warnings")
                st.code(result["stderr"], language="text")
            
            # If we have structured data, display it
            if result.get("structured"):
                st.subheader("Analysis Results")
                
                # Context Guess
                st.markdown("### Context Guess")
                st.write(result["structured"].get('context_guess', 'No context guess available'))
                
                # Confidence
                st.markdown("### Confidence")
                confidence = result["structured"].get('confidence', 0)
                st.progress(confidence)
                st.write(f"{confidence:.2%}")
                
                # Explanation
                st.markdown("### Explanation")
                st.write(result["structured"].get('explanation', 'No explanation available'))
                
                # Related Links
                st.markdown("### Related Links")
                for link in result["structured"].get('related_links', []):
                    st.write(f"- {link}")
                
                # Search Terms
                st.markdown("### Search Terms Used")
                st.write(", ".join(result["structured"].get('search_terms_used', [])))
        else:
            st.error("Analysis failed. Please try again.")

if __name__ == "__main__":
    main()
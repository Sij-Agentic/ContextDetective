import os
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from mcp.types import TextContent

logger = logging.getLogger("ContextDetective.Memory")

class MemoryModule:
    """Handles storage and retrieval of analysis data."""
    
    def __init__(self, storage_path: str = "memory_storage"):
        """Initialize the memory module."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._index = self._load_index()
        logger.info(f"Memory module initialized with storage at {self.storage_path}")
    
    def _load_index(self) -> Dict[str, str]:
        """Load the index of stored analyses."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Index file corrupted, creating new index")
                return {}
        else:
            logger.info("No index file found, creating new index")
            return {}
    
    def _save_index(self):
        """Save the current index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f)
    
    def _compute_image_hash(self, image_path: str) -> str:
        """Compute a hash for an image to uniquely identify it."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_hash = hashlib.md5(image_data).hexdigest()
                logger.debug(f"Computed hash {image_hash} for {image_path}")
                return image_hash
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            return hashlib.md5(image_path.encode()).hexdigest() # Fallback
            
    async def store_analysis(self, image_hash: str, analysis_json: str) -> Dict[str, Any]:
        """
        Store an analysis result.
        
        Args:
            image_hash: Unique hash of the image.
            analysis_json: JSON string containing the complete analysis results.
        """
        try:
            logger.info(f"Storing analysis for image hash {image_hash[:8]}...")
            
            # Validate JSON
            analysis_data = json.loads(analysis_json)
            
            # Save to a file
            entry_path = self.storage_path / f"{image_hash}.json"
            with open(entry_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            # Update index
            self._index[image_hash] = str(entry_path)
            self._save_index()
            
            message = f"Analysis for image {image_hash[:8]} stored successfully."
            logger.info(message)
            return {
                "content": [TextContent(type="text", text=message)]
            }
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON format for analysis data."
            logger.error(error_msg)
            return {
                "content": [TextContent(type="text", text=f"❌ Error: {error_msg}")]
            }
        except Exception as e:
            error_msg = f"Error storing analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }

    async def retrieve_similar_analyses(self, image_hash: str) -> Dict[str, Any]:
        """
        Retrieve similar previous analyses. 
        (Currently retrieves exact match based on hash).
        
        Args:
            image_hash: Hash of the image to retrieve.
        """
        try:
            logger.info(f"Retrieving analysis for image hash {image_hash[:8]}...")
            
            if image_hash not in self._index:
                message = f"No previous analysis found for image hash {image_hash[:8]}."
                logger.info(message)
                return {
                    "content": [TextContent(type="text", text=message)]
                }
            
            entry_path = self._index[image_hash]
            with open(entry_path, 'r') as f:
                analysis_data = json.load(f)
                
            message = f"Retrieved previous analysis for image hash {image_hash[:8]}."
            logger.info(message)
            
            # Return the stored JSON data
            return {
                "content": [TextContent(type="text", text=json.dumps(analysis_data, indent=2))]
            }

        except Exception as e:
            error_msg = f"Error retrieving analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
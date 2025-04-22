import os
import json
import logging
import hashlib
import chromadb
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from mcp.types import TextContent
from chromadb.utils import embedding_functions

logger = logging.getLogger("ContextDetective.Memory")

class MemoryModule:
    """Handles storage and retrieval of analysis data with dual-layer memory system."""
    
    def __init__(self, storage_path: str = "memory_storage", ollama_url: str = "http://localhost:11434"):
        """Initialize the memory module with both file-based and vector storage."""
        # File-based storage (legacy)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.index_file = self.storage_path / "index.json"
        self._index = self._load_index()
        
        # Short-term memory (session-based)
        self.short_term = {}
        
        # Vector database (ChromaDB)
        self.chroma_path = Path(storage_path) / "chromadb"
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Create collections if they don't exist
        self.collections = {
            "complete_analysis": self.chroma_client.get_or_create_collection(
                name="complete_analysis",
                metadata={"description": "Complete analysis results"}
            ),
            "visual_elements": self.chroma_client.get_or_create_collection(
                name="visual_elements",
                metadata={"description": "Visual elements analysis"}
            ),
            "style_analysis": self.chroma_client.get_or_create_collection(
                name="style_analysis", 
                metadata={"description": "Style analysis"}
            ),
            "scenario_analysis": self.chroma_client.get_or_create_collection(
                name="scenario_analysis",
                metadata={"description": "Scenario analysis"}
            )
        }
        
        # Initialize Ollama embedding function
        self.ollama_url = ollama_url
        self.model_name = "mxbai-embed-large"
        
        # Log initialization and memory statistics
        logger.info(f"🚀 Memory module initializing...")
        logger.info(f"   └─ File storage: {self.storage_path}")
        logger.info(f"   └─ ChromaDB: {self.chroma_path}")
        logger.info(f"   └─ Ollama URL: {self.ollama_url}")
        
        # Log statistics by calling our new method
        self.log_memory_statistics()
    
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
            logger.info(f"Computing hash for image at {image_path}")
            
            # Verify file exists first
            if not os.path.exists(image_path):
                logger.error(f"File not found: {image_path}")
                return "error_file_not_found"
            
            # Verify file is readable
            if not os.access(image_path, os.R_OK):
                logger.error(f"File not readable: {image_path}")
                return "error_file_not_readable"
            
            # Get file size for debug info
            file_size = os.path.getsize(image_path)
            logger.info(f"File size: {file_size} bytes")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_hash = hashlib.md5(image_data).hexdigest()
                logger.info(f"Computed hash {image_hash} for {image_path}")
                return image_hash
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            return "error_file_not_found"
        except PermissionError:
            logger.error(f"Permission denied for: {image_path}")
            return "error_permission_denied"
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}", exc_info=True)
            return f"error_{str(e).replace(' ', '_')[:20]}"  # Return error code instead of throwing
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama's mxbai-embed-large model."""
        try:
            url = f"{self.ollama_url}/api/embeddings"
            response = requests.post(
                url,
                json={"model": self.model_name, "prompt": text}
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                logger.error(f"Error generating embedding: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def store_in_short_term(self, session_id: str, component: str, data: Dict[str, Any]) -> None:
        """Store analysis component in short-term memory."""
        if session_id not in self.short_term:
            self.short_term[session_id] = {}
            logger.info(f"🧠 Created new short-term memory session: {session_id[:8]}")
        
        # Get data summary for logging
        data_summary = ""
        if isinstance(data, dict):
            keys = list(data.keys())
            data_summary = f"containing keys: {', '.join(keys[:3])}" + ("..." if len(keys) > 3 else "")
        elif isinstance(data, str):
            data_summary = f"text ({len(data)} chars)"
        else:
            data_summary = f"data of type {type(data).__name__}"
        
        self.short_term[session_id][component] = {
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"📝 SHORT-TERM: Stored '{component}' ({data_summary}) for session {session_id[:8]}")
    
    def get_from_short_term(self, session_id: str, component: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis component from short-term memory."""
        if session_id in self.short_term and component in self.short_term[session_id]:
            timestamp = self.short_term[session_id][component]["timestamp"]
            logger.info(f"🔍 SHORT-TERM: Retrieved '{component}' for session {session_id[:8]} (from {timestamp})")
            return self.short_term[session_id][component]["data"]
        
        logger.info(f"⚠️ SHORT-TERM: Component '{component}' not found for session {session_id[:8]}")
        return None
    
    def clear_short_term(self, session_id: str) -> None:
        """Clear short-term memory for a session."""
        if session_id in self.short_term:
            components = list(self.short_term[session_id].keys())
            del self.short_term[session_id]
            logger.info(f"🧹 SHORT-TERM: Cleared memory for session {session_id[:8]}")
            logger.info(f"   └─ Removed components: {', '.join(components)}")
        else:
            logger.info(f"⚠️ SHORT-TERM: No session {session_id[:8]} to clear")
    
    async def check_exact_match(self, image_hash: str) -> Dict[str, Any]:
        """Check if we have an exact match for this image hash."""
        try:
            logger.info(f"⚡ Checking for exact match for image hash {image_hash[:8]}...")
            
            if image_hash in self._index:
                entry_path = self._index[image_hash]
                if os.path.exists(entry_path):
                    with open(entry_path, 'r') as f:
                        analysis_data = json.load(f)
                    
                    # Enhanced logging with more details
                    original_timestamp = analysis_data.get("timestamp", "unknown time")
                    context = analysis_data.get("context_guess", "unknown context")
                    confidence = analysis_data.get("confidence", "unknown confidence")
                    
                    logger.info(f"🔄 CACHE HIT: Found exact match for image hash {image_hash[:8]}")
                    logger.info(f"   └─ Original analysis from: {original_timestamp}")
                    logger.info(f"   └─ Context: {context}")
                    logger.info(f"   └─ Confidence: {confidence}")
                    
                    return {
                        "match_found": True,
                        "content": [TextContent(type="text", text=json.dumps(analysis_data, indent=2))],
                        "data": analysis_data
                    }
            
            logger.info(f"🔍 CACHE MISS: No exact match found for image hash {image_hash[:8]}")
            return {
                "match_found": False,
                "content": [TextContent(type="text", text=f"No exact match found for image hash {image_hash[:8]}")]
            }
            
        except Exception as e:
            error_msg = f"Error checking for exact match: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {
                "match_found": False,
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }
    
    async def store_analysis(self, image_hash: str, analysis_json: str) -> Dict[str, Any]:
        """
        Store an analysis result in both file storage and vector database.
        
        Args:
            image_hash: Unique hash of the image.
            analysis_json: JSON string containing the complete analysis results.
        """
        try:
            logger.info(f"💾 Storing analysis for image hash {image_hash[:8]}...")
            
            # Parse and validate JSON
            analysis_data = json.loads(analysis_json)
            
            # Add timestamp if not present
            if "timestamp" not in analysis_data:
                analysis_data["timestamp"] = datetime.now().isoformat()
            
            # Extract key information for logging
            context = analysis_data.get("context_guess", "unknown context")
            confidence = analysis_data.get("confidence", "unknown confidence")
            
            # Save to a file (legacy storage)
            entry_path = self.storage_path / f"{image_hash}.json"
            with open(entry_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"📁 LONG-TERM: Saved analysis to file storage: {entry_path}")
            
            # Update index
            self._index[image_hash] = str(entry_path)
            self._save_index()
            logger.info(f"📇 LONG-TERM: Updated index with entry for {image_hash[:8]}")
            
            # Extract components for vector storage
            complete_analysis = json.dumps(analysis_data)
            
            # Generate embeddings for the complete analysis
            logger.info(f"🔢 LONG-TERM: Generating embeddings for analysis...")
            complete_embedding = self._generate_embedding(complete_analysis)
            
            # Store in ChromaDB with appropriate metadata
            if complete_embedding:
                timestamp = datetime.now().isoformat()
                
                # Add to complete_analysis collection
                logger.info(f"🧮 LONG-TERM: Storing in ChromaDB complete_analysis collection...")
                self.collections["complete_analysis"].add(
                    ids=[image_hash],
                    embeddings=[complete_embedding],
                    documents=[complete_analysis],
                    metadatas=[{
                        "image_hash": image_hash,
                        "timestamp": timestamp,
                        "confidence": analysis_data.get("confidence", 0.0),
                        "context": context
                    }]
                )
                
                # Extract and store individual components if available
                components_stored = []
                
                if "visual_elements" in analysis_data:
                    visual_text = json.dumps(analysis_data["visual_elements"])
                    visual_embedding = self._generate_embedding(visual_text)
                    if visual_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB visual_elements collection...")
                        self.collections["visual_elements"].add(
                            ids=[f"{image_hash}_visual"],
                            embeddings=[visual_embedding],
                            documents=[visual_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("visual_elements")
                
                if "style_analysis" in analysis_data:
                    style_text = json.dumps(analysis_data["style_analysis"])
                    style_embedding = self._generate_embedding(style_text)
                    if style_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB style_analysis collection...")
                        self.collections["style_analysis"].add(
                            ids=[f"{image_hash}_style"],
                            embeddings=[style_embedding],
                            documents=[style_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("style_analysis")
                
                if "scenario_analysis" in analysis_data:
                    scenario_text = json.dumps(analysis_data["scenario_analysis"])
                    scenario_embedding = self._generate_embedding(scenario_text)
                    if scenario_embedding:
                        logger.info(f"🧮 LONG-TERM: Storing in ChromaDB scenario_analysis collection...")
                        self.collections["scenario_analysis"].add(
                            ids=[f"{image_hash}_scenario"],
                            embeddings=[scenario_embedding],
                            documents=[scenario_text],
                            metadatas=[{
                                "image_hash": image_hash,
                                "timestamp": timestamp
                            }]
                        )
                        components_stored.append("scenario_analysis")
                
                logger.info(f"✅ LONG-TERM: Vector embeddings stored for collections: complete_analysis, {', '.join(components_stored)}")
            else:
                logger.warning(f"⚠️ LONG-TERM: Failed to generate embeddings for {image_hash[:8]}")
            
            logger.info(f"✅ LONG-TERM: Analysis stored successfully for image {image_hash[:8]}")
            logger.info(f"   └─ Context: {context}")
            logger.info(f"   └─ Confidence: {confidence}")
            logger.info(f"   └─ Location: {entry_path}")
            
            return {
                "content": [TextContent(type="text", text=f"Analysis for image {image_hash[:8]} stored successfully.")]
            }
            
        except json.JSONDecodeError:
            error_msg = "Invalid JSON format for analysis data."
            logger.error(f"❌ LONG-TERM: {error_msg}")
            return {
                "content": [TextContent(type="text", text=f"❌ Error: {error_msg}")]
            }
        except Exception as e:
            error_msg = f"Error storing analysis: {str(e)}"
            logger.error(f"❌ LONG-TERM: {error_msg}", exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }

    async def retrieve_similar_analyses(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve semantically similar analyses based on text query.
        
        Args:
            query_text: Text to use for similarity search
            top_k: Number of results to return
        """
        try:
            logger.info(f"Retrieving similar analyses for query: {query_text[:50]}...")
            
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query_text)
            
            if not query_embedding:
                message = "Failed to generate embedding for query."
                logger.warning(message)
                return {
                    "content": [TextContent(type="text", text=message)]
                }
            
            # Query the complete_analysis collection
            results = self.collections["complete_analysis"].query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results["documents"] or not results["documents"][0]:
                message = "No similar analyses found."
                logger.info(message)
                return {
                    "content": [TextContent(type="text", text=message)]
                }
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results["documents"][0]):
                try:
                    analysis = json.loads(doc)
                    metadata = results["metadatas"][0][i] if results["metadatas"] and i < len(results["metadatas"][0]) else {}
                    similarity = results["distances"][0][i] if results["distances"] and i < len(results["distances"][0]) else None
                    
                    formatted_result = {
                        "analysis": analysis,
                        "metadata": metadata,
                        "similarity": similarity
                    }
                    formatted_results.append(formatted_result)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse document {i}")
            
            message = f"Found {len(formatted_results)} similar analyses."
            logger.info(message)
            
            # Return the formatted results
            return {
                "content": [TextContent(type="text", text=json.dumps(formatted_results, indent=2))],
                "data": formatted_results
            }

        except Exception as e:
            error_msg = f"Error retrieving similar analyses: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")]
            }

    async def retrieve_memory_for_inference(self, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant past analyses to augment inference.
        
        Args:
            current_analysis: Current analysis components
        """
        try:
            logger.info("🧠 Retrieving memory for inference augmentation...")
            
            # Compose query from current analysis components
            query_components = []
            
            if "visual_elements" in current_analysis:
                query_components.append(f"Visual elements: {current_analysis['visual_elements']}")
            
            if "style_analysis" in current_analysis:
                query_components.append(f"Style: {current_analysis['style_analysis']}")
            
            if "scenario_analysis" in current_analysis:
                query_components.append(f"Scenario: {current_analysis['scenario_analysis']}")
            
            query_text = "\n\n".join(query_components)
            logger.debug(f"Generated memory query from {len(query_components)} components")
            
            # Retrieve similar analyses
            result = await self.retrieve_similar_analyses(query_text, top_k=2)
            
            # Format for inference
            if "data" in result and result["data"]:
                inference_context = "RELEVANT PAST ANALYSES:\n\n"
                
                for i, item in enumerate(result["data"]):
                    analysis = item["analysis"]
                    similarity = item["similarity"]
                    metadata = item.get("metadata", {})
                    timestamp = metadata.get("timestamp", "unknown time")
                    
                    logger.info(f"🔍 Found similar analysis #{i+1} (similarity: {similarity:.4f})")
                    logger.info(f"   └─ Image hash: {metadata.get('image_hash', 'unknown')[:8]}")
                    logger.info(f"   └─ Context: {analysis.get('context_guess', 'unknown context')}")
                    logger.info(f"   └─ From: {timestamp}")
                    
                    inference_context += f"--- Analysis {i+1} (Similarity: {similarity:.4f}) ---\n"
                    
                    if "context_guess" in analysis:
                        inference_context += f"Context: {analysis['context_guess']}\n"
                    
                    if "explanation" in analysis:
                        inference_context += f"Explanation: {analysis['explanation'][:300]}...\n"
                    
                    inference_context += "\n"
                
                logger.info(f"✅ Retrieved {len(result['data'])} relevant analyses for inference")
                
                return {
                    "content": [TextContent(type="text", text=inference_context)],
                    "found": True
                }
            else:
                message = "No relevant past analyses found to augment inference."
                logger.info(f"⚠️ {message}")
                return {
                    "content": [TextContent(type="text", text=message)],
                    "found": False
                }
            
        except Exception as e:
            error_msg = f"Error retrieving memory for inference: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {
                "content": [TextContent(type="text", text=f"❌ {error_msg}")],
                "found": False
            }

    def log_memory_statistics(self):
        """Log statistics about both short-term and long-term memory."""
        try:
            # Short-term memory stats
            session_count = len(self.short_term)
            total_components = sum(len(session) for session in self.short_term.values())
            
            # Long-term memory stats
            file_count = len(self._index)
            
            # ChromaDB stats
            complete_count = len(self.collections["complete_analysis"].get()["ids"])
            visual_count = len(self.collections["visual_elements"].get()["ids"])
            style_count = len(self.collections["style_analysis"].get()["ids"])
            scenario_count = len(self.collections["scenario_analysis"].get()["ids"])
            
            logger.info(f"📊 MEMORY STATISTICS:")
            logger.info(f"   └─ SHORT-TERM: {session_count} active sessions with {total_components} total components")
            logger.info(f"   └─ LONG-TERM File: {file_count} stored analyses")
            logger.info(f"   └─ LONG-TERM Vector: {complete_count} complete analyses")
            logger.info(f"   └─ LONG-TERM Vector Components: {visual_count} visual, {style_count} style, {scenario_count} scenario")
            
            return {
                "short_term": {
                    "sessions": session_count,
                    "components": total_components
                },
                "long_term": {
                    "file_storage": file_count,
                    "vector_storage": {
                        "complete": complete_count,
                        "visual": visual_count,
                        "style": style_count,
                        "scenario": scenario_count
                    }
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting memory statistics: {str(e)}", exc_info=True)
            return {
                "error": str(e)
            }
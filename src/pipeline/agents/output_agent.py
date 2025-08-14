"""Output agent for the SQL generation pipeline.

This agent handles writing the final SQL query and metadata to files,
creating backups, and managing output formatting.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_agent import BasePipelineAgent


class OutputAgent(BasePipelineAgent):
    """
    Agent responsible for output generation in the pipeline.
    
    This agent handles:
    - Writing final SQL queries to files
    - Creating backup files
    - Generating metadata files
    - Formatting output according to specifications
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="output_agent")
    
    def _get_agent_tools(self) -> List:
        """Get output-related tools for the agent (none needed for file operations)."""
        return []
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute output generation step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - final_query: The final SQL query to output
                - translated_query: Optional translated query
                - pipeline_metadata: Metadata about the pipeline execution
                - original_prompt: Original prompt for reference
            config: Configuration parameters for this step including:
                - output_file: Target output file path
                - output_directory: Output directory path
                - backup_existing: Whether to backup existing files
                - include_metadata: Whether to include metadata file
                - timestamp_files: Whether to add timestamps to filenames
                
        Returns:
            Dictionary containing output operation results
        """
        try:
            # Extract input data
            final_query = input_data.get("final_query", "")
            translated_query = input_data.get("translated_query", "")
            pipeline_metadata = input_data.get("pipeline_metadata", {})
            original_prompt = input_data.get("original_prompt", "")
            
            # Use translated query if available, otherwise use final_query
            query_to_output = translated_query if translated_query and translated_query.strip() else final_query
            
            # Extract configuration
            output_file = config.get("output_file", None)
            output_directory = config.get("output_directory", "./output")
            backup_existing = config.get("backup_existing", True)
            include_metadata = config.get("include_metadata", True)
            timestamp_files = config.get("timestamp_files", True)
            
            # Validate inputs
            if not query_to_output or not query_to_output.strip():
                return self._create_error_result("No SQL query provided for output")
            
            # Determine output file path
            output_path = self._determine_output_path(
                output_file, output_directory, timestamp_files
            )
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists and backup is enabled
            backup_path = None
            if output_path.exists() and backup_existing:
                backup_path = self._create_backup(output_path)
            
            # Write the SQL query
            write_result = self._write_sql_file(
                output_path, query_to_output, original_prompt, pipeline_metadata
            )
            
            if not write_result.get("success", False):
                return write_result
            
            # Write metadata file if requested
            metadata_path = None
            if include_metadata:
                metadata_path = self._write_metadata_file(
                    output_path, pipeline_metadata, original_prompt, query_to_output
                )
            
            # Generate output summary
            output_summary = self._generate_output_summary(
                output_path, query_to_output, backup_path, metadata_path
            )
            
            return self._create_success_result({
                "output_file_path": str(output_path),
                "backup_file_path": str(backup_path) if backup_path else None,
                "metadata_file_path": str(metadata_path) if metadata_path else None,
                "query_length": len(query_to_output),
                "backup_created": backup_path is not None,
                "metadata_written": metadata_path is not None,
                "output_summary": output_summary
            })
        
        except Exception as e:
            return self._create_error_result(f"Output generation failed: {e}")
    
    def _determine_output_path(self, output_file: Optional[str], output_directory: str, 
                             timestamp_files: bool) -> Path:
        """Determine the final output file path."""
        try:
            output_dir = Path(output_directory)
            
            if output_file:
                # Use specified output file
                if Path(output_file).is_absolute():
                    output_path = Path(output_file)
                else:
                    output_path = output_dir / output_file
            else:
                # Generate default filename
                if timestamp_files:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_query_{timestamp}.sql"
                else:
                    filename = "generated_query.sql"
                
                output_path = output_dir / filename
            
            return output_path
            
        except Exception as e:
            # Fallback to simple filename
            return Path(output_directory) / "generated_query.sql"
    
    def _create_backup(self, output_path: Path) -> Optional[Path]:
        """Create a backup of existing file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{output_path.stem}_backup_{timestamp}{output_path.suffix}"
            backup_path = output_path.parent / backup_name
            
            shutil.copy2(output_path, backup_path)
            self._log_info(f"Created backup: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            self._log_error(f"Failed to create backup: {e}")
            return None
    
    def _write_sql_file(self, output_path: Path, sql_query: str, original_prompt: str, 
                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Write the SQL query to file with formatting."""
        try:
            # Format the SQL file content
            file_content = self._format_sql_output(sql_query, original_prompt, metadata)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            file_size = output_path.stat().st_size
            
            self._log_info(f"SQL query written to: {output_path} ({file_size} bytes)")
            
            return self._create_success_result({
                "file_path": str(output_path),
                "file_size": file_size,
                "lines_written": file_content.count('\\n') + 1
            })
            
        except Exception as e:
            return self._create_error_result(f"Failed to write SQL file: {e}")
    
    def _write_metadata_file(self, sql_output_path: Path, metadata: Dict[str, Any], 
                           original_prompt: str, sql_query: str) -> Optional[Path]:
        """Write metadata file alongside the SQL file."""
        try:
            metadata_path = sql_output_path.with_suffix('.json')
            
            # Prepare metadata content
            metadata_content = {
                "generation_info": {
                    "timestamp": datetime.now().isoformat(),
                    "sql_file": str(sql_output_path),
                    "original_prompt": original_prompt,
                    "query_length": len(sql_query),
                    "query_lines": sql_query.count('\\n') + 1
                },
                "pipeline_metadata": metadata,
                "file_info": {
                    "created_by": "NOVASQLAgent Pipeline",
                    "version": "1.0.0",
                    "format": "JSON"
                }
            }
            
            # Write metadata file
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_content, f, indent=2, ensure_ascii=False)
            
            self._log_info(f"Metadata written to: {metadata_path}")
            
            return metadata_path
            
        except Exception as e:
            self._log_error(f"Failed to write metadata file: {e}")
            return None
    
    def _format_sql_output(self, sql_query: str, original_prompt: str, 
                         metadata: Dict[str, Any]) -> str:
        """Format the SQL output with header comments."""
        try:
            # Create header comment
            header_lines = [
                "-- ==================================================",
                "-- Generated SQL Query",
                "-- ==================================================",
                f"-- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "-- Generated by: NOVASQLAgent Pipeline",
                ""
            ]
            
            # Add original prompt as comment
            if original_prompt:
                header_lines.extend([
                    "-- Original Request:",
                    f"-- {original_prompt}",
                    ""
                ])
            
            # Add pipeline info from metadata
            if metadata:
                pipeline_info = metadata.get("pipeline_info", {})
                if pipeline_info:
                    header_lines.extend([
                        "-- Pipeline Information:",
                        f"-- Steps completed: {pipeline_info.get('steps_completed', 'unknown')}",
                        f"-- Processing time: {pipeline_info.get('total_duration', 'unknown')}s",
                        ""
                    ])
            
            header_lines.extend([
                "-- ==================================================",
                "",
                ""
            ])
            
            # Combine header and SQL query
            formatted_content = "\\n".join(header_lines) + sql_query
            
            # Ensure SQL ends with semicolon and newline
            if not formatted_content.rstrip().endswith(';'):
                formatted_content = formatted_content.rstrip() + ';'
            
            formatted_content += "\\n"
            
            return formatted_content
            
        except Exception as e:
            self._log_error(f"Failed to format SQL output: {e}")
            # Return query as-is if formatting fails
            return sql_query
    
    def _generate_output_summary(self, output_path: Path, sql_query: str, 
                               backup_path: Optional[Path], metadata_path: Optional[Path]) -> Dict[str, Any]:
        """Generate a summary of the output operation."""
        try:
            summary = {
                "output_file": {
                    "path": str(output_path),
                    "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                    "exists": output_path.exists()
                },
                "query_info": {
                    "length": len(sql_query),
                    "lines": sql_query.count('\\n') + 1,
                    "has_comments": "--" in sql_query,
                    "ends_with_semicolon": sql_query.rstrip().endswith(';')
                },
                "files_created": {
                    "sql_file": output_path.exists(),
                    "backup_file": backup_path.exists() if backup_path else False,
                    "metadata_file": metadata_path.exists() if metadata_path else False
                }
            }
            
            if backup_path:
                summary["backup_file"] = {
                    "path": str(backup_path),
                    "size_bytes": backup_path.stat().st_size if backup_path.exists() else 0
                }
            
            if metadata_path:
                summary["metadata_file"] = {
                    "path": str(metadata_path),
                    "size_bytes": metadata_path.stat().st_size if metadata_path.exists() else 0
                }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to generate summary: {e}"}
    
    def _validate_inputs(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs specific to output agent."""
        try:
            # Call parent validation first
            parent_result = super()._validate_inputs(input_data, config)
            if not parent_result["valid"]:
                return parent_result
            
            # Check for required SQL query
            final_query = input_data.get("final_query", "")
            translated_query = input_data.get("translated_query", "")
            
            if not final_query and not translated_query:
                return {"valid": False, "error": "No SQL query provided for output"}
            
            # Validate output directory
            output_directory = config.get("output_directory", "./output")
            try:
                Path(output_directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return {"valid": False, "error": f"Cannot create output directory: {e}"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Input validation failed: {e}"}
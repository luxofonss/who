import json
import datetime
from pathlib import Path
from typing import Any, Union, Optional, Dict
from loguru import logger


PathLike = Union[str, Path]


def ensure_dir(path: PathLike):
    """Create the directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: PathLike, default: Any | None = None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return default


def write_json(path: PathLike, data: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 


def read_file(filepath: PathLike, encoding: str = "utf-8", default: str = "") -> str:
    """
    Read a file and return its contents as a string.
    
    Args:
        filepath: Path to the file to read
        encoding: File encoding (default: utf-8)
        default: Default value to return if file cannot be read
        
    Returns:
        File contents as string, or default value if file cannot be read
    """
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            logger.warning(f"File does not exist: {filepath}")
            return default
            
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
            logger.debug(f"Successfully read file: {filepath} ({len(content)} characters)")
            return content
            
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return default
    except PermissionError:
        logger.error(f"Permission denied reading file: {filepath}")
        return default
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error reading file {filepath}: {e}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error reading file {filepath}: {e}")
        return default


def write_analysis_results(state_data: Dict[str, Any], project_id: str, endpoint: str, base_dir: str = "storage/analyze") -> Dict[str, str]:
    """
    Write analysis results from state to multiple file formats.
    
    Args:
        state_data: The state dictionary containing analysis results
        project_id: Project identifier
        endpoint: API endpoint being analyzed
        base_dir: Base directory for storing analysis files
        
    Returns:
        Dictionary with file paths for each written file
    """
    try:
        # Create timestamp for unique file names
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize endpoint for filename
        safe_endpoint = endpoint.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
        if len(safe_endpoint) > 50:
            safe_endpoint = safe_endpoint[:50]
        
        # Create base directory
        analysis_dir = Path(base_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file paths
        json_file = analysis_dir / f"{project_id}_{safe_endpoint}_{timestamp}.json"
        html_file = analysis_dir / f"{project_id}_{safe_endpoint}_{timestamp}.html"
        summary_file = analysis_dir / f"{project_id}_{safe_endpoint}_{timestamp}_summary.txt"
        
        written_files = {}
        
        # Write JSON data
        try:
            write_json(json_file, state_data)
            written_files["json"] = str(json_file)
            logger.info(f"✅ Written JSON analysis to: {json_file}")
        except Exception as e:
            logger.error(f"❌ Failed to write JSON file: {e}")
        
        # Write HTML content if available
        if state_data.get("html_response"):
            try:
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(state_data["html_response"])
                written_files["html"] = str(html_file)
                logger.info(f"✅ Written HTML analysis to: {html_file}")
            except Exception as e:
                logger.error(f"❌ Failed to write HTML file: {e}")
        
        # Write summary text file
        try:
            summary_content = _build_analysis_summary(state_data, project_id, endpoint, timestamp)
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            written_files["summary"] = str(summary_file)
            logger.info(f"✅ Written analysis summary to: {summary_file}")
        except Exception as e:
            logger.error(f"❌ Failed to write summary file: {e}")
        
        return written_files
        
    except Exception as e:
        logger.error(f"❌ Error writing analysis results: {e}")
        return {}


def _build_analysis_summary(state_data: Dict[str, Any], project_id: str, endpoint: str, timestamp: str) -> str:
    """Build a human-readable summary of the analysis results."""
    
    summary_parts = [
        f"Analysis Report",
        f"=" * 50,
        f"Project ID: {project_id}",
        f"Endpoint: {endpoint}",
        f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timestamp: {timestamp}",
        f"",
    ]
    
    # Add phase completion status
    if state_data.get("phase_complete"):
        summary_parts.append("Phase Completion Status:")
        for phase, completed in state_data["phase_complete"].items():
            status = "✅ Completed" if completed else "❌ Not completed"
            summary_parts.append(f"  - {phase}: {status}")
        summary_parts.append("")
    
    # Add test cases summary
    if state_data.get("existing_testcases"):
        summary_parts.append(f"Existing Test Cases: {len(state_data['existing_testcases'])}")
    
    if state_data.get("generated_missing_testcases"):
        summary_parts.append(f"Generated Missing Test Cases: {len(state_data['generated_missing_testcases'])}")
    
    if state_data.get("final_testcases"):
        summary_parts.append(f"Final Test Cases: {len(state_data['final_testcases'])}")
    
    # Add acceptance criteria summary
    if state_data.get("current_ac"):
        summary_parts.append(f"Current Acceptance Criteria: {len(state_data['current_ac'])}")
    
    if state_data.get("generated_missing_ac"):
        summary_parts.append(f"Generated Missing AC: {len(state_data['generated_missing_ac'])}")
    
    if state_data.get("final_ac"):
        summary_parts.append(f"Final Acceptance Criteria: {len(state_data['final_ac'])}")
    
    # Add coverage information
    if state_data.get("additional_coverage"):
        coverage = state_data["additional_coverage"]
        summary_parts.append("")
        summary_parts.append("Coverage Analysis:")
        if coverage.get("coverage_summary"):
            summary_parts.append(f"  Summary: {coverage['coverage_summary']}")
        if coverage.get("coverage_metrics"):
            metrics = coverage["coverage_metrics"]
            for metric, value in metrics.items():
                summary_parts.append(f"  {metric}: {value:.2f}%")
    
    # Add final analysis result
    if state_data.get("final_analysis_result"):
        result = state_data["final_analysis_result"]
        summary_parts.append("")
        summary_parts.append("Final Analysis Result:")
        if result.get("overall_coverage"):
            summary_parts.append(f"  Overall Coverage: {result['overall_coverage']:.2f}%")
        if result.get("test_coverage"):
            summary_parts.append(f"  Test Coverage: {result['test_coverage']:.2f}%")
        if result.get("ac_coverage"):
            summary_parts.append(f"  AC Coverage: {result['ac_coverage']:.2f}%")
    
    # Add context information
    if state_data.get("context"):
        summary_parts.append("")
        summary_parts.append(f"Context Length: {len(state_data['context'])} characters")
    
    if state_data.get("requirements"):
        req_length = len(state_data["requirements"])
        summary_parts.append(f"Requirements Length: {req_length} characters")
        if req_length > 2000:
            summary_parts.append("  (truncated in summary)")
    
    return "\n".join(summary_parts)


def _write_to_file(content: str, prefix: str, path: PathLike) -> str:
    """Write content to a file with timestamp and metadata."""
    try:
        prompts_dir = Path(path)
        prompts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.txt"
        filepath = prompts_dir / filename
        
        logger.debug(f"Writing to file: {filepath}")
        logger.debug(f"Content length: {len(content)} characters")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Type: {prefix}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        logger.info(f"Successfully wrote to file: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to write to file: {e}")
        logger.error(f"Path: {path}, Prefix: {prefix}, Content length: {len(content)}")
        return ""

def _split_csv_field(field):
    if not field:
        return []
    return [x.strip() for x in field.split(",") if x.strip()]

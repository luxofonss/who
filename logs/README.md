# Analysis Logs

This directory contains debug files generated during LangGraph analysis workflows.

## Directory Structure

- **`prompts/`** - Contains all prompts sent to the LLM during analysis
- **`responses/`** - Contains all responses received from the LLM during analysis

## File Naming Convention

### Prompts
- `agent_iteration_{N}_{project_id}_{timestamp}.txt` - Agent prompts for each iteration
- `fallback_analysis_{project_id}_{timestamp}.txt` - Fallback analysis prompts

### Responses  
- `response_iteration_{N}_{project_id}_{timestamp}.txt` - Agent responses for each iteration
- `response_iteration_0_{project_id}_{timestamp}.txt` - Fallback analysis responses (iteration 0)

## File Format

Each file contains:
- Header with project ID, iteration number, and timestamp
- Separator line (`=======`)
- Full prompt or response content

## Usage

These files are automatically generated during analysis and can be used for:
- Debugging prompt engineering
- Understanding agent reasoning flow
- Reviewing LLM responses
- Analyzing workflow performance

## Cleanup

Files are automatically gitignored and can be safely deleted when no longer needed for debugging. 
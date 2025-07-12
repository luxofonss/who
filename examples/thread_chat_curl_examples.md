# Thread Chat API - cURL Examples

This document provides cURL examples for using the new thread chat API endpoint.

## Prerequisites

1. Make sure your FastAPI server is running on `http://localhost:8000`
2. Ensure you have a valid project ID (e.g., `onestudy-server-demo`)
3. Create a thread first using the threads API

## API Endpoints

### 1. Create a Thread

```bash
curl -X POST "http://localhost:8000/threads" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Discussion Thread",
    "description": "Thread for discussing API implementation details",
    "project_id": "onestudy-server-demo",
    "branch": "main"
  }'
```

**Response:**
```json
{
  "status": "created",
  "thread": {
    "thread_id": "thread_a1b2c3d4",
    "name": "API Discussion Thread",
    "description": "Thread for discussing API implementation details",
    "project_id": "onestudy-server-demo",
    "branch": "main",
    "is_active": true,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "last_activity": "2024-01-15T10:30:00Z"
  }
}
```

### 2. Send Chat Message

```bash
curl -X POST "http://localhost:8000/threads/thread_a1b2c3d4/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main API endpoints in this project?"
  }'
```

**Response:**
```json
{
  "thread_id": "thread_a1b2c3d4",
  "user_message": {
    "message_id": "msg_e5f6g7h8",
    "content": "What are the main API endpoints in this project?",
    "role": "user",
    "created_at": "2024-01-15T10:35:00Z"
  },
  "ai_response": {
    "message_id": "msg_i9j0k1l2",
    "content": "Based on the codebase analysis, this project has several main API endpoints...",
    "role": "assistant",
    "created_at": "2024-01-15T10:35:05Z",
    "analysis_result": {
      "method": "langgraph",
      "iterations": 2,
      "symbols_retrieved": ["UserController", "AuthService"],
      "context_used": "Retrieved controller and service classes..."
    }
  }
}
```

### 3. Get Thread Messages

```bash
curl -X GET "http://localhost:8000/threads/thread_a1b2c3d4/messages?limit=10&offset=0"
```

**Response:**
```json
{
  "thread_id": "thread_a1b2c3d4",
  "messages": [
    {
      "id": 2,
      "message_id": "msg_i9j0k1l2",
      "thread_id": "thread_a1b2c3d4",
      "role": "assistant",
      "content": "Based on the codebase analysis, this project has several main API endpoints...",
      "analysis_result": {
        "method": "langgraph",
        "iterations": 2,
        "symbols_retrieved": ["UserController", "AuthService"],
        "context_used": "Retrieved controller and service classes..."
      },
      "created_at": "2024-01-15T10:35:05Z",
      "updated_at": "2024-01-15T10:35:05Z"
    },
    {
      "id": 1,
      "message_id": "msg_e5f6g7h8",
      "thread_id": "thread_a1b2c3d4",
      "role": "user",
      "content": "What are the main API endpoints in this project?",
      "analysis_result": null,
      "created_at": "2024-01-15T10:35:00Z",
      "updated_at": "2024-01-15T10:35:00Z"
    }
  ],
  "pagination": {
    "total": 2,
    "limit": 10,
    "offset": 0,
    "current_page": 1,
    "total_pages": 1,
    "has_next": false,
    "has_previous": false
  }
}
```

## Complete Example Workflow

Here's a complete workflow example:

```bash
# 1. Create a thread
THREAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/threads" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Code Analysis Thread",
    "description": "Thread for analyzing codebase",
    "project_id": "onestudy-server-demo",
    "branch": "main"
  }')

# Extract thread ID
THREAD_ID=$(echo $THREAD_RESPONSE | jq -r '.thread.thread_id')
echo "Created thread: $THREAD_ID"

# 2. Send multiple chat messages
curl -X POST "http://localhost:8000/threads/$THREAD_ID/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How does authentication work in this project?"}'

curl -X POST "http://localhost:8000/threads/$THREAD_ID/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What database entities are used?"}'

# 3. Get all messages
curl -X GET "http://localhost:8000/threads/$THREAD_ID/messages?limit=20"
```

## Error Handling

### Thread Not Found
```bash
curl -X POST "http://localhost:8000/threads/invalid_thread_id/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

**Response:**
```json
{
  "detail": "Thread not found"
}
```

### Inactive Thread
```bash
curl -X POST "http://localhost:8000/threads/inactive_thread_id/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

**Response:**
```json
{
  "detail": "Thread is not active"
}
```

### Invalid Message
```bash
curl -X POST "http://localhost:8000/threads/thread_a1b2c3d4/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": ""}'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

## Features

- **Conversation Context**: The AI maintains conversation history and context
- **Code Analysis**: Uses ChatChain with LangGraph for intelligent code analysis
- **Message Persistence**: All messages are stored in the database
- **Analysis Metadata**: Each AI response includes analysis details (method, iterations, symbols retrieved)
- **Thread Management**: Messages are organized by threads with activity tracking
- **Pagination**: Support for retrieving messages with pagination 
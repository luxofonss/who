#!/usr/bin/env python3
"""
Example script demonstrating how to use the thread chat API endpoint
"""

import requests
import json
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your server runs on a different port

def create_thread(project_id: str, name: str, description: str = None) -> Dict[str, Any]:
    """Create a new thread for a project"""
    url = f"{BASE_URL}/threads"
    data = {
        "name": name,
        "description": description,
        "project_id": project_id,
        "branch": "main"
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def send_chat_message(thread_id: str, message: str) -> Dict[str, Any]:
    """Send a chat message to a thread and get AI response"""
    url = f"{BASE_URL}/threads/{thread_id}/chat"
    data = {
        "message": message
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()

def get_thread_messages(thread_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get latest messages from a thread"""
    url = f"{BASE_URL}/threads/{thread_id}/messages"
    params = {"limit": limit}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    """Example usage of the thread chat API"""
    print("🤖 Thread Chat API Example")
    print("=" * 50)
    
    # Example project ID (replace with an actual project ID from your system)
    project_id = "onestudy-server-demo"  # This should be an existing project
    
    try:
        # Step 1: Create a new thread
        print("\n📝 Creating a new thread...")
        thread_response = create_thread(
            project_id=project_id,
            name="API Discussion Thread",
            description="Thread for discussing API implementation details"
        )
        
        thread_id = thread_response["thread"]["thread_id"]
        print(f"✅ Thread created with ID: {thread_id}")
        
        # Step 2: Send some chat messages
        messages = [
            "What are the main API endpoints in this project?",
            "How does the authentication system work?",
            "Can you explain the UserService implementation?",
            "What database entities are used for user management?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n💬 Sending message {i}: {message[:50]}...")
            
            chat_response = send_chat_message(thread_id, message)
            
            print(f"🤖 AI Response: {chat_response['ai_response']['content'][:100]}...")
            print(f"📊 Analysis: {chat_response['ai_response']['analysis_result']['method']} method, "
                  f"{chat_response['ai_response']['analysis_result']['iterations']} iterations")
        
        # Step 3: Get thread messages
        print(f"\n📋 Getting latest messages from thread {thread_id}...")
        messages_response = get_thread_messages(thread_id, limit=5)
        
        print(f"📊 Found {len(messages_response['messages'])} messages:")
        for msg in messages_response['messages']:
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            print(f"{role_emoji} {msg['role'].title()}: {msg['content'][:80]}...")
        
        print("\n✅ Example completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
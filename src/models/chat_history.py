"""
Chat History models for storing conversation messages in threads
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from .database import Base

class ChatHistory(Base):
    """
    Chat History model for storing conversation messages in project threads
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String(255), unique=True, index=True, nullable=False)
    thread_id = Column(String(255), ForeignKey("project_threads.thread_id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)  # The message content
    analysis_result = Column(JSON, nullable=True)  # JSON result from analysis
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, message_id='{self.message_id}', thread_id='{self.thread_id}', role='{self.role}')>"

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = {
            "id": self.id,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "analysis_result": self.analysis_result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        return data 
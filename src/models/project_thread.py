"""
Project Thread models for conversation context management (flat version)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from .database import Base

class ProjectThread(Base):
    """
    Project Thread model - simplified: one API endpoint, documents and jira links as comma-separated strings
    """
    __tablename__ = "project_threads"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    project_id = Column(String(255), ForeignKey("projects.project_id"), nullable=False)
    branch = Column(String(100), default="main")
    context_summary = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    message_count = Column(Integer, default=0)
    last_activity = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # New fields for flat storage
    api_method = Column(String(10), nullable=True)  # GET, POST, etc.
    api_path = Column(String(500), nullable=True)   # /api/users
    documents = Column(Text, nullable=True)         # Comma-separated business doc URLs
    api_documents = Column(Text, nullable=True)     # Comma-separated API doc URLs
    jira_links = Column(Text, nullable=True)        # Comma-separated URLs
    references = Column(Text, nullable=True)        # Comma-separated class/method symbols

    def __repr__(self):
        return f"<ProjectThread(id={self.id}, thread_id='{self.thread_id}', name='{self.name}')>"

    def to_dict(self, include_details=False):
        """Convert to dictionary for JSON serialization"""
        data = {
            "id": self.id,
            "thread_id": self.thread_id,
            "name": self.name,
            "description": self.description,
            "project_id": self.project_id,
            "branch": self.branch,
            "context_summary": self.context_summary,
            "is_active": self.is_active,
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "api_method": self.api_method,
            "api_path": self.api_path,
            "documents": self.documents,
            "api_documents": self.api_documents,
            "jira_links": self.jira_links,
            "references": self.references,
        }
        return data 
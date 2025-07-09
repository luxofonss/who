"""
Project model for storing project information
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from .database import Base


class Project(Base):
    """
    Project model for storing project information
    """
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Bitbucket information
    bitbucket_url = Column(String(500), nullable=False)
    workspace = Column(String(255), nullable=False)
    repository = Column(String(255), nullable=False)
    default_branch = Column(String(100), default="main")
    
    # Project metadata
    commit_hash = Column(String(100), nullable=True)
    indexed_files = Column(Integer, default=0)
    extracted_files = Column(Integer, default=0)
    dependency_graph = Column(JSON, nullable=True)
    
    # Status and tracking
    status = Column(String(50), default="active")  # active, archived, error
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_indexed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Project(id={self.id}, project_id='{self.project_id}', name='{self.name}')>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "bitbucket_url": self.bitbucket_url,
            "workspace": self.workspace,
            "repository": self.repository,
            "default_branch": self.default_branch,
            "commit_hash": self.commit_hash,
            "indexed_files": self.indexed_files,
            "extracted_files": self.extracted_files,
            "status": self.status,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        } 
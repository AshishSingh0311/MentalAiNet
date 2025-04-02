"""
User management module for MH-Net.

This module provides functionalities for user authentication, authorization, 
and session management for clinicians and administrators.
"""

import os
import json
import hashlib
import secrets
import datetime
import time
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from typing import Dict, List, Optional, Union, Any

Base = declarative_base()

# Define user roles association table
user_roles = Table(
    'user_roles', 
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


class User(Base):
    """User model for clinicians and administrators."""
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(32), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    activity_logs = relationship("ActivityLog", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
    
    def to_dict(self):
        """Convert user object to dictionary without sensitive information."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "roles": [role.name for role in self.roles]
        }


class Role(Base):
    """Role model for access control."""
    
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", back_populates="role", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Role(name='{self.name}')>"
    
    def to_dict(self):
        """Convert role object to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": [perm.to_dict() for perm in self.permissions]
        }


class Permission(Base):
    """Permission model for fine-grained access control."""
    
    __tablename__ = 'permissions'
    
    id = Column(Integer, primary_key=True)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    resource = Column(String(50), nullable=False)  # e.g., 'patient', 'assessment'
    action = Column(String(50), nullable=False)  # e.g., 'view', 'edit', 'delete'
    
    # Relationship
    role = relationship("Role", back_populates="permissions")
    
    def __repr__(self):
        return f"<Permission(resource='{self.resource}', action='{self.action}')>"
    
    def to_dict(self):
        """Convert permission object to dictionary."""
        return {
            "id": self.id,
            "resource": self.resource,
            "action": self.action
        }


class Session(Base):
    """User session model for tracking active sessions."""
    
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_token = Column(String(64), unique=True, nullable=False)
    ip_address = Column(String(50))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<Session(user_id='{self.user_id}', expires_at='{self.expires_at}')>"
    
    def to_dict(self):
        """Convert session object to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active
        }


class ActivityLog(Base):
    """Activity log model for audit trail."""
    
    __tablename__ = 'activity_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    action = Column(String(50), nullable=False)  # e.g., 'login', 'view_patient', 'edit_assessment'
    resource_id = Column(Integer)  # ID of the resource being acted upon (e.g., patient_id)
    resource_type = Column(String(50))  # Type of resource (e.g., 'patient', 'assessment')
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    ip_address = Column(String(50))
    details = Column(Text)  # JSON string of additional details
    
    # Relationship
    user = relationship("User", back_populates="activity_logs")
    
    def __repr__(self):
        return f"<ActivityLog(user_id='{self.user_id}', action='{self.action}')>"
    
    def to_dict(self):
        """Convert activity log object to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "ip_address": self.ip_address,
            "details": json.loads(self.details) if self.details else None
        }


class UserManager:
    """Manager class for handling user operations."""
    
    def __init__(self, db_url):
        """Initialize the UserManager with a database connection."""
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)  # Create tables if they don't exist
        
        # Initialize default roles if they don't exist
        self._initialize_roles()
    
    def _initialize_roles(self):
        """Initialize default roles if they don't exist."""
        session = self.Session()
        
        # Check if roles exist
        if session.query(Role).count() == 0:
            # Create default roles
            admin_role = Role(name="admin", description="Administrator with full access")
            clinician_role = Role(name="clinician", description="Clinician with patient access")
            researcher_role = Role(name="researcher", description="Researcher with limited access")
            
            # Add admin permissions
            admin_permissions = [
                Permission(resource="user", action="view"),
                Permission(resource="user", action="create"),
                Permission(resource="user", action="edit"),
                Permission(resource="user", action="delete"),
                Permission(resource="patient", action="view"),
                Permission(resource="patient", action="create"),
                Permission(resource="patient", action="edit"),
                Permission(resource="patient", action="delete"),
                Permission(resource="assessment", action="view"),
                Permission(resource="assessment", action="create"),
                Permission(resource="assessment", action="edit"),
                Permission(resource="assessment", action="delete"),
                Permission(resource="model", action="view"),
                Permission(resource="model", action="train"),
                Permission(resource="model", action="deploy"),
                Permission(resource="report", action="view"),
                Permission(resource="report", action="create"),
                Permission(resource="report", action="export"),
                Permission(resource="dashboard", action="view"),
                Permission(resource="system", action="configure")
            ]
            
            # Add clinician permissions
            clinician_permissions = [
                Permission(resource="patient", action="view"),
                Permission(resource="patient", action="create"),
                Permission(resource="patient", action="edit"),
                Permission(resource="assessment", action="view"),
                Permission(resource="assessment", action="create"),
                Permission(resource="assessment", action="edit"),
                Permission(resource="model", action="view"),
                Permission(resource="report", action="view"),
                Permission(resource="report", action="create"),
                Permission(resource="report", action="export"),
                Permission(resource="dashboard", action="view")
            ]
            
            # Add researcher permissions
            researcher_permissions = [
                Permission(resource="patient", action="view"),
                Permission(resource="assessment", action="view"),
                Permission(resource="model", action="view"),
                Permission(resource="model", action="train"),
                Permission(resource="report", action="view"),
                Permission(resource="report", action="create"),
                Permission(resource="report", action="export"),
                Permission(resource="dashboard", action="view")
            ]
            
            # Add permissions to roles
            admin_role.permissions = admin_permissions
            clinician_role.permissions = clinician_permissions
            researcher_role.permissions = researcher_permissions
            
            # Add roles to database
            session.add_all([admin_role, clinician_role, researcher_role])
            session.commit()
        
        session.close()
    
    def create_user(self, username, email, password, first_name=None, last_name=None, 
                   roles=None, is_active=True, is_verified=False):
        """
        Create a new user.
        
        Args:
            username (str): Username
            email (str): Email address
            password (str): Password
            first_name (str, optional): First name
            last_name (str, optional): Last name
            roles (list, optional): List of role names
            is_active (bool, optional): Whether the user is active
            is_verified (bool, optional): Whether the user is verified
            
        Returns:
            dict: User information if successful, None otherwise
        """
        session = self.Session()
        
        try:
            # Check if username or email already exists
            if session.query(User).filter(User.username == username).first():
                return {"error": "Username already exists"}
            
            if session.query(User).filter(User.email == email).first():
                return {"error": "Email already exists"}
            
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)
            
            # Create new user
            new_user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                first_name=first_name,
                last_name=last_name,
                is_active=is_active,
                is_verified=is_verified
            )
            
            # Add roles if provided
            if roles:
                for role_name in roles:
                    role = session.query(Role).filter(Role.name == role_name).first()
                    if role:
                        new_user.roles.append(role)
            
            # Add user to database
            session.add(new_user)
            session.commit()
            
            # Return user information
            return new_user.to_dict()
        except Exception as e:
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()
    
    def authenticate_user(self, username_or_email, password):
        """
        Authenticate a user.
        
        Args:
            username_or_email (str): Username or email
            password (str): Password
            
        Returns:
            dict: User information and session token if successful, None otherwise
        """
        session = self.Session()
        
        try:
            # Find user by username or email
            user = session.query(User).filter(
                (User.username == username_or_email) | (User.email == username_or_email)
            ).first()
            
            if not user:
                return {"error": "User not found"}
            
            if not user.is_active:
                return {"error": "User is inactive"}
            
            # Verify password
            password_hash = self._hash_password(password, user.salt)
            if password_hash != user.password_hash:
                return {"error": "Invalid password"}
            
            # Update last login time
            user.last_login = datetime.datetime.utcnow()
            
            # Create new session
            session_token = secrets.token_hex(32)
            expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=1)
            
            new_session = Session(
                user_id=user.id,
                session_token=session_token,
                expires_at=expires_at
            )
            
            session.add(new_session)
            session.commit()
            
            # Return user information and session token
            return {
                "user": user.to_dict(),
                "session_token": session_token,
                "expires_at": expires_at.isoformat()
            }
        except Exception as e:
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()
    
    def verify_session(self, session_token):
        """
        Verify a session token.
        
        Args:
            session_token (str): Session token
            
        Returns:
            dict: User information if successful, None otherwise
        """
        db_session = self.Session()
        
        try:
            # Find session by token
            session = db_session.query(Session).filter(
                Session.session_token == session_token,
                Session.is_active == True,
                Session.expires_at > datetime.datetime.utcnow()
            ).first()
            
            if not session:
                return {"error": "Invalid or expired session"}
            
            # Get user
            user = db_session.query(User).filter(User.id == session.user_id).first()
            
            if not user or not user.is_active:
                return {"error": "User not found or inactive"}
            
            # Return user information
            return {"user": user.to_dict()}
        except Exception as e:
            return {"error": str(e)}
        finally:
            db_session.close()
    
    def end_session(self, session_token):
        """
        End a session.
        
        Args:
            session_token (str): Session token
            
        Returns:
            bool: True if successful, False otherwise
        """
        db_session = self.Session()
        
        try:
            # Find session by token
            session = db_session.query(Session).filter(
                Session.session_token == session_token
            ).first()
            
            if not session:
                return False
            
            # Set session as inactive
            session.is_active = False
            db_session.commit()
            
            return True
        except Exception as e:
            db_session.rollback()
            return False
        finally:
            db_session.close()
    
    def has_permission(self, user_id, resource, action):
        """
        Check if a user has permission to perform an action on a resource.
        
        Args:
            user_id (int): User ID
            resource (str): Resource name
            action (str): Action name
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        db_session = self.Session()
        
        try:
            # Get user with roles
            user = db_session.query(User).filter(User.id == user_id).first()
            
            if not user or not user.is_active:
                return False
            
            # Check permissions for each role
            for role in user.roles:
                for permission in role.permissions:
                    if permission.resource == resource and permission.action == action:
                        return True
            
            return False
        except Exception as e:
            return False
        finally:
            db_session.close()
    
    def log_activity(self, user_id, action, resource_id=None, resource_type=None, 
                    ip_address=None, details=None):
        """
        Log user activity.
        
        Args:
            user_id (int): User ID
            action (str): Action name
            resource_id (int, optional): Resource ID
            resource_type (str, optional): Resource type
            ip_address (str, optional): IP address
            details (dict, optional): Additional details
            
        Returns:
            bool: True if successful, False otherwise
        """
        db_session = self.Session()
        
        try:
            # Create activity log
            new_log = ActivityLog(
                user_id=user_id,
                action=action,
                resource_id=resource_id,
                resource_type=resource_type,
                ip_address=ip_address,
                details=json.dumps(details) if details else None
            )
            
            db_session.add(new_log)
            db_session.commit()
            
            return True
        except Exception as e:
            db_session.rollback()
            return False
        finally:
            db_session.close()
    
    def update_user(self, user_id, **kwargs):
        """
        Update user information.
        
        Args:
            user_id (int): User ID
            **kwargs: Fields to update
            
        Returns:
            dict: Updated user information if successful, None otherwise
        """
        db_session = self.Session()
        
        try:
            # Get user
            user = db_session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {"error": "User not found"}
            
            # Update fields
            for field, value in kwargs.items():
                if field == 'password':
                    # Generate new salt and hash password
                    salt = secrets.token_hex(16)
                    password_hash = self._hash_password(value, salt)
                    user.password_hash = password_hash
                    user.salt = salt
                elif field == 'roles':
                    # Update roles
                    user.roles = []
                    for role_name in value:
                        role = db_session.query(Role).filter(Role.name == role_name).first()
                        if role:
                            user.roles.append(role)
                elif hasattr(user, field):
                    setattr(user, field, value)
            
            db_session.commit()
            
            # Return updated user information
            return user.to_dict()
        except Exception as e:
            db_session.rollback()
            return {"error": str(e)}
        finally:
            db_session.close()
    
    def delete_user(self, user_id):
        """
        Delete a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        db_session = self.Session()
        
        try:
            # Get user
            user = db_session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return False
            
            # Delete user
            db_session.delete(user)
            db_session.commit()
            
            return True
        except Exception as e:
            db_session.rollback()
            return False
        finally:
            db_session.close()
    
    def get_user(self, user_id):
        """
        Get user information.
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: User information if successful, None otherwise
        """
        db_session = self.Session()
        
        try:
            # Get user
            user = db_session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            # Return user information
            return user.to_dict()
        except Exception as e:
            return None
        finally:
            db_session.close()
    
    def get_users(self, filter_params=None):
        """
        Get all users with optional filtering.
        
        Args:
            filter_params (dict, optional): Filter parameters
            
        Returns:
            list: List of user information
        """
        db_session = self.Session()
        
        try:
            # Start with base query
            query = db_session.query(User)
            
            # Apply filters if provided
            if filter_params:
                for field, value in filter_params.items():
                    if hasattr(User, field):
                        query = query.filter(getattr(User, field) == value)
            
            # Execute query
            users = query.all()
            
            # Return user information
            return [user.to_dict() for user in users]
        except Exception as e:
            return []
        finally:
            db_session.close()
    
    def get_roles(self):
        """
        Get all roles.
        
        Returns:
            list: List of role information
        """
        db_session = self.Session()
        
        try:
            # Get all roles
            roles = db_session.query(Role).all()
            
            # Return role information
            return [role.to_dict() for role in roles]
        except Exception as e:
            return []
        finally:
            db_session.close()
    
    def _hash_password(self, password, salt):
        """
        Hash password with salt.
        
        Args:
            password (str): Password
            salt (str): Salt
            
        Returns:
            str: Hashed password
        """
        # Combine password and salt
        salted_password = password + salt
        
        # Hash the combined string
        hash_obj = hashlib.sha256(salted_password.encode())
        return hash_obj.hexdigest()


# Simplified interface for user management
def get_user_manager(db_url):
    """
    Get a UserManager instance.
    
    Args:
        db_url (str): Database URL
        
    Returns:
        UserManager: UserManager instance
    """
    return UserManager(db_url)
import os
import datetime
import json
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create a base class for declarative class definitions
Base = declarative_base()

# Define the Patient model
class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(String(50), unique=True, nullable=False)
    age = Column(Integer)
    gender = Column(String(20))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with assessments
    assessments = relationship("Assessment", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient(patient_id='{self.patient_id}', age={self.age})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "created_at": self.created_at.isoformat()
        }

# Define the Assessment model
class Assessment(Base):
    __tablename__ = 'assessments'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    clinician = Column(String(100))
    session_type = Column(String(50))
    assessment_date = Column(String(20))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    risk_scores = Column(JSON)
    text_input = Column(Text, nullable=True)
    has_audio = Column(Boolean, default=False)
    has_physio = Column(Boolean, default=False)
    has_imaging = Column(Boolean, default=False)
    
    # Relationship with patient
    patient = relationship("Patient", back_populates="assessments")
    
    def __repr__(self):
        return f"<Assessment(id={self.id}, patient_id={self.patient_id})>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "clinician": self.clinician,
            "session_type": self.session_type,
            "assessment_date": self.assessment_date,
            "timestamp": self.timestamp.isoformat(),
            "risk_scores": self.risk_scores,
            "text_input": self.text_input,
            "has_audio": self.has_audio,
            "has_physio": self.has_physio,
            "has_imaging": self.has_imaging
        }

class DatabaseManager:
    """Handles database operations for the MH-Net framework."""
    
    def __init__(self):
        """Initialize the database connection using environment variables."""
        self.db_url = os.environ.get('DATABASE_URL', 'sqlite:///mhnet.db')
        # Create engine with appropriate configuration
        self.engine = create_engine(
            self.db_url,
            pool_pre_ping=True,  # Enable connection health checks
            pool_recycle=3600,   # Recycle connections after 1 hour
            pool_size=5,         # Connection pool size
            max_overflow=10      # Max additional connections
        )
        
        # Create all tables if they don't exist
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"Warning: Could not create database tables: {e}")
        
        # Create a sessionmaker
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.Session()
    
    def add_patient(self, patient_data):
        """
        Add a new patient to the database.
        
        Args:
            patient_data: Dictionary containing patient information
        
        Returns:
            Newly created patient object
        """
        session = self.get_session()
        try:
            # Check if patient already exists
            existing_patient = session.query(Patient).filter_by(
                patient_id=patient_data['patient_id']
            ).first()
            
            if existing_patient:
                return existing_patient
            
            # Create new patient
            patient = Patient(
                patient_id=patient_data['patient_id'],
                age=patient_data['age'],
                gender=patient_data['gender']
            )
            
            session.add(patient)
            session.commit()
            return patient
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_assessment(self, assessment_data):
        """
        Add a new assessment to the database.
        
        Args:
            assessment_data: Dictionary containing assessment information
        
        Returns:
            Newly created assessment object
        """
        session = self.get_session()
        try:
            # Get patient ID
            patient = session.query(Patient).filter_by(
                patient_id=assessment_data['patient_id']
            ).first()
            
            if not patient:
                # Create patient if not exists
                patient = Patient(
                    patient_id=assessment_data['patient_id'],
                    age=assessment_data['metadata']['age'],
                    gender=assessment_data['metadata']['gender']
                )
                session.add(patient)
                session.flush()
            
            # Create assessment
            assessment = Assessment(
                patient_id=patient.id,
                clinician=assessment_data['metadata']['clinician'],
                session_type=assessment_data['metadata']['session_type'],
                assessment_date=assessment_data['metadata']['assessment_date'],
                risk_scores=assessment_data['risk_scores'],
                text_input=assessment_data['text_input'],
                has_audio=assessment_data['has_audio'],
                has_physio=assessment_data['has_physio'],
                has_imaging=assessment_data['has_imaging']
            )
            
            session.add(assessment)
            session.commit()
            return assessment
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_patient(self, patient_id):
        """
        Get a patient by patient_id.
        
        Args:
            patient_id: The patient_id to search for
        
        Returns:
            Patient object if found, None otherwise
        """
        session = self.get_session()
        try:
            return session.query(Patient).filter_by(patient_id=patient_id).first()
        finally:
            session.close()
    
    def get_all_patients(self):
        """
        Get all patients.
        
        Returns:
            List of all patients
        """
        session = self.get_session()
        try:
            return session.query(Patient).all()
        except Exception as e:
            print(f"Error getting all patients: {e}")
            # Return empty list on error instead of failing
            return []
        finally:
            session.close()
    
    def get_assessments_for_patient(self, patient_id):
        """
        Get all assessments for a patient.
        
        Args:
            patient_id: The patient_id to get assessments for
        
        Returns:
            List of assessments for the patient
        """
        session = self.get_session()
        try:
            patient = session.query(Patient).filter_by(patient_id=patient_id).first()
            if patient:
                return session.query(Assessment).filter_by(patient_id=patient.id).order_by(Assessment.timestamp.desc()).all()
            return []
        except Exception as e:
            print(f"Error getting assessments for patient {patient_id}: {e}")
            return []
        finally:
            session.close()
    
    def get_all_assessments(self):
        """
        Get all assessments.
        
        Returns:
            List of all assessments
        """
        session = self.get_session()
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return session.query(Assessment).order_by(Assessment.timestamp.desc()).all()
            except Exception as e:
                retry_count += 1
                print(f"Error getting all assessments (attempt {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    # Return empty list after max retries instead of failing
                    print("Max retries reached, returning empty assessment list")
                    return []
                # Wait before retrying with exponential backoff
                time.sleep(0.5 * (2 ** retry_count))
            finally:
                if retry_count >= max_retries:
                    session.close()
    
    def get_assessment(self, assessment_id):
        """
        Get an assessment by ID.
        
        Args:
            assessment_id: The assessment ID to retrieve
        
        Returns:
            Assessment object if found, None otherwise
        """
        session = self.get_session()
        try:
            return session.query(Assessment).filter_by(id=assessment_id).first()
        finally:
            session.close()
    
    def delete_assessment(self, assessment_id):
        """
        Delete an assessment.
        
        Args:
            assessment_id: The assessment ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session()
        try:
            assessment = session.query(Assessment).filter_by(id=assessment_id).first()
            if assessment:
                session.delete(assessment)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Create database manager instance
db_manager = DatabaseManager()
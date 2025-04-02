"""
Batch processor module for MH-Net.

This module provides functionality for batch processing of patient assessments
and model predictions.
"""

import os
import json
import pandas as pd
import numpy as np
import time
import datetime
import tempfile
import threading
import queue
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from uuid import uuid4

from modules.database import DatabaseManager


class BatchJob:
    """Class representing a batch processing job."""
    
    def __init__(self, job_id: str, job_type: str, params: Dict[str, Any], 
                description: str = None, created_by: str = None):
        """
        Initialize a batch job.
        
        Args:
            job_id (str): Unique identifier for the job
            job_type (str): Type of job (e.g., "assessment", "prediction")
            params (dict): Parameters for the job
            description (str, optional): Job description
            created_by (str, optional): User who created the job
        """
        self.job_id = job_id
        self.job_type = job_type
        self.params = params
        self.description = description
        self.created_by = created_by
        self.status = "pending"
        self.created_at = datetime.datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.results = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the job to a dictionary.
        
        Returns:
            dict: Dictionary representation of the job
        """
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "params": self.params,
            "description": self.description,
            "created_by": self.created_by,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "error": self.error
        }


class BatchProcessor:
    """Class for handling batch processing of patient assessments and predictions."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, 
                max_workers: int = 2, job_dir: Optional[str] = None):
        """
        Initialize the BatchProcessor.
        
        Args:
            db_manager (DatabaseManager, optional): Database manager for storing and retrieving data
            max_workers (int): Maximum number of worker threads
            job_dir (str, optional): Directory for storing job data
        """
        self.db_manager = db_manager
        self.max_workers = max_workers
        
        # Set job directory
        if job_dir:
            self.job_dir = job_dir
            os.makedirs(self.job_dir, exist_ok=True)
        else:
            self.job_dir = tempfile.mkdtemp(prefix="mhnet_jobs_")
        
        # Initialize job storage
        self.jobs = {}
        self.job_queue = queue.Queue()
        self.running = True
        
        # Start worker threads
        self.workers = []
        for _ in range(max_workers):
            worker = threading.Thread(target=self._worker_thread, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def create_job(self, job_type: str, params: Dict[str, Any], 
                  description: str = None, created_by: str = None) -> str:
        """
        Create a new batch job.
        
        Args:
            job_type (str): Type of job (e.g., "assessment", "prediction")
            params (dict): Parameters for the job
            description (str, optional): Job description
            created_by (str, optional): User who created the job
            
        Returns:
            str: Job ID
        """
        # Generate a unique job ID
        job_id = str(uuid4())
        
        # Create job
        job = BatchJob(job_id, job_type, params, description, created_by)
        
        # Store job
        self.jobs[job_id] = job
        
        # Save job to disk
        self._save_job(job)
        
        # Add job to queue
        self.job_queue.put(job_id)
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by ID.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            dict: Job information if found, None otherwise
        """
        if job_id in self.jobs:
            return self.jobs[job_id].to_dict()
        else:
            # Try to load from disk
            job = self._load_job(job_id)
            if job:
                self.jobs[job_id] = job
                return job.to_dict()
            return None
    
    def get_jobs(self, status: Optional[str] = None, 
                job_type: Optional[str] = None,
                created_by: Optional[str] = None,
                limit: int = 100, 
                offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get a list of jobs with optional filtering.
        
        Args:
            status (str, optional): Filter by status
            job_type (str, optional): Filter by job type
            created_by (str, optional): Filter by creator
            limit (int): Maximum number of jobs to return
            offset (int): Number of jobs to skip
            
        Returns:
            list: List of job dictionaries
        """
        # Get all jobs from memory and disk
        all_jobs = list(self.jobs.values())
        
        # Load jobs from disk that aren't in memory
        job_files = [f for f in os.listdir(self.job_dir) if f.endswith(".json")]
        for job_file in job_files:
            job_id = job_file.split(".")[0]
            if job_id not in self.jobs:
                job = self._load_job(job_id)
                if job:
                    all_jobs.append(job)
        
        # Apply filters
        filtered_jobs = []
        for job in all_jobs:
            if status and job.status != status:
                continue
            if job_type and job.job_type != job_type:
                continue
            if created_by and job.created_by != created_by:
                continue
            filtered_jobs.append(job)
        
        # Sort by creation time (newest first)
        filtered_jobs.sort(key=lambda j: j.created_at if j.created_at else datetime.datetime.min, reverse=True)
        
        # Apply pagination
        paginated_jobs = filtered_jobs[offset:offset+limit]
        
        # Convert to dictionaries
        return [job.to_dict() for job in paginated_jobs]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if job_id not in self.jobs:
            # Try to load from disk
            job = self._load_job(job_id)
            if job:
                self.jobs[job_id] = job
            else:
                return False
        
        job = self.jobs[job_id]
        
        # Can only cancel pending or running jobs
        if job.status not in ["pending", "running"]:
            return False
        
        # Update status
        job.status = "cancelled"
        job.completed_at = datetime.datetime.now()
        job.error = "Job cancelled by user"
        
        # Save job to disk
        self._save_job(job)
        
        return True
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if job_id not in self.jobs:
            # Try to load from disk
            job = self._load_job(job_id)
            if not job:
                return False
        
        # Remove from memory
        if job_id in self.jobs:
            del self.jobs[job_id]
        
        # Remove from disk
        job_path = os.path.join(self.job_dir, f"{job_id}.json")
        if os.path.exists(job_path):
            os.remove(job_path)
        
        # Remove results if they exist
        results_path = os.path.join(self.job_dir, f"{job_id}_results.json")
        if os.path.exists(results_path):
            os.remove(results_path)
        
        return True
    
    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the results of a completed job.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            dict: Job results if available, None otherwise
        """
        job = self.get_job(job_id)
        
        if not job or job["status"] != "completed":
            return None
        
        # Try to load results from disk
        results_path = os.path.join(self.job_dir, f"{job_id}_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # If job is in memory, return results
        if job_id in self.jobs and self.jobs[job_id].results:
            return self.jobs[job_id].results
        
        return None
    
    def export_job_results(self, job_id: str, format: str = "json") -> Optional[str]:
        """
        Export job results to a file.
        
        Args:
            job_id (str): Job ID
            format (str): Export format (json, csv)
            
        Returns:
            str: Path to exported file if successful, None otherwise
        """
        results = self.get_job_results(job_id)
        
        if not results:
            return None
        
        # Create export file path
        export_path = os.path.join(self.job_dir, f"{job_id}_export.{format}")
        
        try:
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format == "csv":
                # Convert results to DataFrame
                if isinstance(results, list):
                    pd.DataFrame(results).to_csv(export_path, index=False)
                elif isinstance(results, dict):
                    if "items" in results and isinstance(results["items"], list):
                        pd.DataFrame(results["items"]).to_csv(export_path, index=False)
                    else:
                        pd.DataFrame([results]).to_csv(export_path, index=False)
                else:
                    return None
            else:
                return None
            
            return export_path
        except:
            return None
    
    def update_job_progress(self, job_id: str, progress: float, 
                           message: Optional[str] = None) -> bool:
        """
        Update job progress.
        
        Args:
            job_id (str): Job ID
            progress (float): Progress value (0-1)
            message (str, optional): Progress message
            
        Returns:
            bool: True if successful, False otherwise
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Update progress
        job.progress = max(0.0, min(1.0, progress))
        
        # Save job to disk
        self._save_job(job)
        
        return True
    
    def _worker_thread(self):
        """Worker thread for processing jobs."""
        while self.running:
            try:
                # Get a job from the queue
                job_id = self.job_queue.get(timeout=1)
                
                # Get the job
                if job_id in self.jobs:
                    job = self.jobs[job_id]
                    
                    # Skip cancelled jobs
                    if job.status == "cancelled":
                        self.job_queue.task_done()
                        continue
                    
                    # Update job status
                    job.status = "running"
                    job.started_at = datetime.datetime.now()
                    self._save_job(job)
                    
                    try:
                        # Process the job based on its type
                        if job.job_type == "assessment_batch":
                            self._process_assessment_batch(job)
                        elif job.job_type == "prediction_batch":
                            self._process_prediction_batch(job)
                        elif job.job_type == "import_batch":
                            self._process_import_batch(job)
                        elif job.job_type == "export_batch":
                            self._process_export_batch(job)
                        else:
                            raise ValueError(f"Unknown job type: {job.job_type}")
                        
                        # Update job status
                        job.status = "completed"
                        job.completed_at = datetime.datetime.now()
                        job.progress = 1.0
                    except Exception as e:
                        # Update job status on error
                        job.status = "failed"
                        job.completed_at = datetime.datetime.now()
                        job.error = str(e)
                    
                    # Save job to disk
                    self._save_job(job)
                
                self.job_queue.task_done()
            except queue.Empty:
                # No jobs available
                pass
            except Exception as e:
                # Log the error
                print(f"Worker thread error: {e}")
    
    def _process_assessment_batch(self, job: BatchJob):
        """
        Process a batch of assessments.
        
        Args:
            job (BatchJob): Batch job
        """
        params = job.params
        
        # Check required parameters
        if "assessments" not in params or not isinstance(params["assessments"], list):
            raise ValueError("Missing or invalid 'assessments' parameter")
        
        assessments = params["assessments"]
        total = len(assessments)
        results = []
        
        for i, assessment_data in enumerate(assessments):
            try:
                # Process each assessment
                if self.db_manager:
                    # Add to database
                    assessment = self.db_manager.add_assessment(assessment_data)
                    
                    if assessment:
                        results.append({
                            "success": True,
                            "assessment_id": assessment.id,
                            "patient_id": assessment.patient_id,
                            "timestamp": assessment.timestamp.isoformat() if hasattr(assessment.timestamp, 'isoformat') else str(assessment.timestamp)
                        })
                    else:
                        results.append({
                            "success": False,
                            "error": "Failed to add assessment to database",
                            "assessment_data": assessment_data
                        })
                else:
                    # Simulate processing without database
                    time.sleep(0.1)
                    
                    # Add a fake assessment ID
                    results.append({
                        "success": True,
                        "assessment_id": f"a-{uuid4()}",
                        "patient_id": assessment_data.get("patient_id", "unknown"),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            except Exception as e:
                # Handle individual assessment failure
                results.append({
                    "success": False,
                    "error": str(e),
                    "assessment_data": assessment_data
                })
            
            # Update progress
            progress = (i + 1) / total
            self.update_job_progress(job.job_id, progress)
        
        # Store results
        job.results = {
            "total": total,
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "items": results
        }
        
        # Save results to disk
        results_path = os.path.join(self.job_dir, f"{job.job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(job.results, f, indent=2)
    
    def _process_prediction_batch(self, job: BatchJob):
        """
        Process a batch of predictions.
        
        Args:
            job (BatchJob): Batch job
        """
        params = job.params
        
        # Check required parameters
        if "inputs" not in params or not isinstance(params["inputs"], list):
            raise ValueError("Missing or invalid 'inputs' parameter")
        
        inputs = params["inputs"]
        total = len(inputs)
        results = []
        
        # Get model configuration
        model_config = params.get("model_config", {})
        
        for i, input_data in enumerate(inputs):
            try:
                # Simulate prediction
                time.sleep(0.2)
                
                # Generate fake prediction results
                conditions = ["Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia"]
                
                prediction = {
                    "input_id": input_data.get("id", f"input_{i}"),
                    "prediction_id": f"p-{uuid4()}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "scores": {}
                }
                
                # Generate random scores that sum to around 1.0
                raw_scores = np.random.random(len(conditions))
                norm_scores = raw_scores / raw_scores.sum()
                
                for j, condition in enumerate(conditions):
                    prediction["scores"][condition] = round(float(norm_scores[j]), 4)
                
                # Add primary condition
                max_condition = max(prediction["scores"].items(), key=lambda x: x[1])
                prediction["primary_condition"] = max_condition[0]
                prediction["primary_score"] = max_condition[1]
                
                results.append({
                    "success": True,
                    "input": input_data,
                    "prediction": prediction
                })
            except Exception as e:
                # Handle individual prediction failure
                results.append({
                    "success": False,
                    "error": str(e),
                    "input": input_data
                })
            
            # Update progress
            progress = (i + 1) / total
            self.update_job_progress(job.job_id, progress)
        
        # Store results
        job.results = {
            "total": total,
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "model_config": model_config,
            "items": results
        }
        
        # Save results to disk
        results_path = os.path.join(self.job_dir, f"{job.job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(job.results, f, indent=2)
    
    def _process_import_batch(self, job: BatchJob):
        """
        Process a batch import job.
        
        Args:
            job (BatchJob): Batch job
        """
        params = job.params
        
        # Check required parameters
        if "data" not in params or not isinstance(params["data"], list):
            raise ValueError("Missing or invalid 'data' parameter")
        
        if "import_type" not in params:
            raise ValueError("Missing 'import_type' parameter")
        
        import_type = params["import_type"]
        data = params["data"]
        total = len(data)
        results = []
        
        for i, item in enumerate(data):
            try:
                # Process based on import type
                if import_type == "patients":
                    if self.db_manager:
                        # Add patient to database
                        patient = self.db_manager.add_patient(item)
                        
                        if patient:
                            results.append({
                                "success": True,
                                "patient_id": patient.id
                            })
                        else:
                            results.append({
                                "success": False,
                                "error": "Failed to add patient to database",
                                "data": item
                            })
                    else:
                        # Simulate processing
                        time.sleep(0.1)
                        results.append({
                            "success": True,
                            "patient_id": f"p-{uuid4()}"
                        })
                elif import_type == "assessments":
                    if self.db_manager:
                        # Add assessment to database
                        assessment = self.db_manager.add_assessment(item)
                        
                        if assessment:
                            results.append({
                                "success": True,
                                "assessment_id": assessment.id
                            })
                        else:
                            results.append({
                                "success": False,
                                "error": "Failed to add assessment to database",
                                "data": item
                            })
                    else:
                        # Simulate processing
                        time.sleep(0.1)
                        results.append({
                            "success": True,
                            "assessment_id": f"a-{uuid4()}"
                        })
                else:
                    raise ValueError(f"Unsupported import type: {import_type}")
            except Exception as e:
                # Handle individual item failure
                results.append({
                    "success": False,
                    "error": str(e),
                    "data": item
                })
            
            # Update progress
            progress = (i + 1) / total
            self.update_job_progress(job.job_id, progress)
        
        # Store results
        job.results = {
            "import_type": import_type,
            "total": total,
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "items": results
        }
        
        # Save results to disk
        results_path = os.path.join(self.job_dir, f"{job.job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(job.results, f, indent=2)
    
    def _process_export_batch(self, job: BatchJob):
        """
        Process a batch export job.
        
        Args:
            job (BatchJob): Batch job
        """
        params = job.params
        
        # Check required parameters
        if "export_type" not in params:
            raise ValueError("Missing 'export_type' parameter")
        
        export_type = params["export_type"]
        format = params.get("format", "json")
        query_params = params.get("query_params", {})
        
        results = []
        
        try:
            # Process based on export type
            if export_type == "patients":
                if self.db_manager:
                    # Get patients from database
                    patients = self.db_manager.get_all_patients()
                    
                    if patients:
                        # Convert patients to dictionaries
                        patient_data = [p.to_dict() for p in patients]
                        
                        # Export to file
                        export_path = os.path.join(self.job_dir, f"{job.job_id}_patients.{format}")
                        
                        if format == "json":
                            with open(export_path, 'w') as f:
                                json.dump(patient_data, f, indent=2)
                        elif format == "csv":
                            pd.DataFrame(patient_data).to_csv(export_path, index=False)
                        else:
                            raise ValueError(f"Unsupported export format: {format}")
                        
                        # Add to results
                        results = patient_data
                else:
                    # Simulate processing
                    time.sleep(0.5)
                    
                    # Generate fake patient data
                    patient_data = []
                    for i in range(10):
                        patient_data.append({
                            "id": i + 1,
                            "patient_id": f"P-{1000 + i}",
                            "age": np.random.randint(18, 80),
                            "gender": np.random.choice(["Male", "Female", "Non-binary"]),
                            "created_at": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 100))).isoformat()
                        })
                    
                    results = patient_data
            elif export_type == "assessments":
                if self.db_manager:
                    # Get assessments from database
                    assessments = self.db_manager.get_all_assessments()
                    
                    if assessments:
                        # Convert assessments to dictionaries
                        assessment_data = [a.to_dict() for a in assessments]
                        
                        # Export to file
                        export_path = os.path.join(self.job_dir, f"{job.job_id}_assessments.{format}")
                        
                        if format == "json":
                            with open(export_path, 'w') as f:
                                json.dump(assessment_data, f, indent=2)
                        elif format == "csv":
                            pd.DataFrame(assessment_data).to_csv(export_path, index=False)
                        else:
                            raise ValueError(f"Unsupported export format: {format}")
                        
                        # Add to results
                        results = assessment_data
                else:
                    # Simulate processing
                    time.sleep(0.5)
                    
                    # Generate fake assessment data
                    assessment_data = []
                    for i in range(20):
                        assessment_data.append({
                            "id": i + 1,
                            "patient_id": np.random.randint(1, 11),
                            "clinician": f"Dr. Smith",
                            "session_type": np.random.choice(["Initial", "Follow-up", "Crisis"]),
                            "assessment_date": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d"),
                            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=np.random.randint(0, 30))).isoformat(),
                            "risk_scores": {
                                "Depression": round(np.random.random() * 0.5, 2),
                                "Anxiety": round(np.random.random() * 0.5, 2),
                                "PTSD": round(np.random.random() * 0.3, 2)
                            }
                        })
                    
                    results = assessment_data
            else:
                raise ValueError(f"Unsupported export type: {export_type}")
            
            # Update progress
            self.update_job_progress(job.job_id, 1.0)
        except Exception as e:
            # Handle export failure
            job.error = str(e)
            job.status = "failed"
            self._save_job(job)
            return
        
        # Store results
        job.results = {
            "export_type": export_type,
            "format": format,
            "count": len(results),
            "query_params": query_params,
            "items": results
        }
        
        # Save results to disk
        results_path = os.path.join(self.job_dir, f"{job.job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(job.results, f, indent=2)
    
    def _save_job(self, job: BatchJob):
        """
        Save a job to disk.
        
        Args:
            job (BatchJob): Batch job
        """
        job_path = os.path.join(self.job_dir, f"{job.job_id}.json")
        
        # Save job info (without results)
        job_info = job.to_dict()
        
        # Don't save results in the job file to avoid duplication
        if "results" in job_info:
            del job_info["results"]
        
        with open(job_path, 'w') as f:
            json.dump(job_info, f, indent=2)
    
    def _load_job(self, job_id: str) -> Optional[BatchJob]:
        """
        Load a job from disk.
        
        Args:
            job_id (str): Job ID
            
        Returns:
            BatchJob: Loaded job if successful, None otherwise
        """
        job_path = os.path.join(self.job_dir, f"{job_id}.json")
        
        if not os.path.exists(job_path):
            return None
        
        try:
            with open(job_path, 'r') as f:
                job_info = json.load(f)
            
            # Create BatchJob from info
            job = BatchJob(
                job_id=job_info["job_id"],
                job_type=job_info["job_type"],
                params=job_info["params"],
                description=job_info.get("description"),
                created_by=job_info.get("created_by")
            )
            
            # Set other attributes
            job.status = job_info.get("status", "pending")
            job.created_at = datetime.datetime.fromisoformat(job_info["created_at"]) if job_info.get("created_at") else None
            job.started_at = datetime.datetime.fromisoformat(job_info["started_at"]) if job_info.get("started_at") else None
            job.completed_at = datetime.datetime.fromisoformat(job_info["completed_at"]) if job_info.get("completed_at") else None
            job.progress = job_info.get("progress", 0.0)
            job.error = job_info.get("error")
            
            return job
        except:
            return None
    
    def shutdown(self):
        """Shutdown the batch processor."""
        self.running = False
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)


class ModelVersionManager:
    """Class for managing model versions."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the ModelVersionManager.
        
        Args:
            models_dir (str, optional): Directory for storing model versions
        """
        # Set models directory
        if models_dir:
            self.models_dir = models_dir
            os.makedirs(self.models_dir, exist_ok=True)
        else:
            self.models_dir = tempfile.mkdtemp(prefix="mhnet_models_")
        
        # Initialize model registry
        self.registry_path = os.path.join(self.models_dir, "registry.json")
        self.registry = self._load_registry()
    
    def register_model(self, name: str, version: str, metadata: Dict[str, Any],
                      model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a new model version.
        
        Args:
            name (str): Model name
            version (str): Model version
            metadata (dict): Model metadata
            model_path (str, optional): Path to model file or directory
            
        Returns:
            dict: Model information
        """
        # Check if version already exists
        model_id = f"{name}/{version}"
        
        if model_id in self.registry:
            raise ValueError(f"Model version already exists: {model_id}")
        
        # Create model info
        model_info = {
            "id": model_id,
            "name": name,
            "version": version,
            "created_at": datetime.datetime.now().isoformat(),
            "metadata": metadata,
            "status": "registered"
        }
        
        # Copy model file if provided
        if model_path:
            model_dir = os.path.join(self.models_dir, name, version)
            os.makedirs(model_dir, exist_ok=True)
            
            if os.path.isfile(model_path):
                # Copy file
                import shutil
                model_filename = os.path.basename(model_path)
                shutil.copy2(model_path, os.path.join(model_dir, model_filename))
                model_info["model_file"] = model_filename
            elif os.path.isdir(model_path):
                # Copy directory contents
                import shutil
                for item in os.listdir(model_path):
                    s = os.path.join(model_path, item)
                    d = os.path.join(model_dir, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
                    else:
                        shutil.copytree(s, d, dirs_exist_ok=True)
                model_info["model_dir"] = True
        
        # Add to registry
        self.registry[model_id] = model_info
        self._save_registry()
        
        return model_info
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a model version.
        
        Args:
            name (str): Model name
            version (str, optional): Model version. If None, the latest version is returned.
            
        Returns:
            dict: Model information if found, None otherwise
        """
        if version:
            # Get specific version
            model_id = f"{name}/{version}"
            return self.registry.get(model_id)
        else:
            # Get latest version
            versions = []
            
            for model_id, model_info in self.registry.items():
                if model_info["name"] == name:
                    versions.append(model_info)
            
            if not versions:
                return None
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda m: m["created_at"], reverse=True)
            
            return versions[0]
    
    def list_models(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models.
        
        Args:
            name (str, optional): Filter by model name
            
        Returns:
            list: List of model information
        """
        models = []
        
        for model_id, model_info in self.registry.items():
            if name and model_info["name"] != name:
                continue
            
            models.append(model_info)
        
        # Sort by name and creation time
        models.sort(key=lambda m: (m["name"], m["created_at"]), reverse=True)
        
        return models
    
    def update_model_status(self, name: str, version: str, 
                           status: str, message: Optional[str] = None) -> bool:
        """
        Update model status.
        
        Args:
            name (str): Model name
            version (str): Model version
            status (str): New status
            message (str, optional): Status message
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_id = f"{name}/{version}"
        
        if model_id not in self.registry:
            return False
        
        # Update status
        self.registry[model_id]["status"] = status
        
        if message:
            self.registry[model_id]["status_message"] = message
        
        self._save_registry()
        
        return True
    
    def delete_model(self, name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            name (str): Model name
            version (str): Model version
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_id = f"{name}/{version}"
        
        if model_id not in self.registry:
            return False
        
        # Remove model directory
        model_dir = os.path.join(self.models_dir, name, version)
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        
        # Remove from registry
        del self.registry[model_id]
        self._save_registry()
        
        return True
    
    def compare_models(self, models: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Compare multiple model versions.
        
        Args:
            models (list): List of (name, version) tuples
            
        Returns:
            dict: Comparison results
        """
        comparison = {
            "models": [],
            "metrics": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        for name, version in models:
            model_info = self.get_model(name, version)
            
            if not model_info:
                continue
            
            # Add to comparison
            comparison["models"].append(model_info)
            
            # Extract metrics from metadata
            metadata = model_info.get("metadata", {})
            metrics = metadata.get("metrics", {})
            
            for metric_name, metric_value in metrics.items():
                if metric_name not in comparison["metrics"]:
                    comparison["metrics"][metric_name] = []
                
                comparison["metrics"][metric_name].append({
                    "model_id": model_info["id"],
                    "value": metric_value
                })
        
        return comparison
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the model registry from disk.
        
        Returns:
            dict: Model registry
        """
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {}
    
    def _save_registry(self):
        """Save the model registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
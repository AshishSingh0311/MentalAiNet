"""
API integration module for MH-Net.

This module provides functionality for integrating with external APIs
and electronic health record systems.
"""

import requests
import json
import os
import time
import hashlib
import hmac
import base64
import urllib.parse
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable


class APIClient:
    """Base class for API clients."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize the API client.
        
        Args:
            base_url (str): Base URL for the API
            api_key (str, optional): API key for authentication
            timeout (int): Request timeout in seconds
            verify_ssl (bool): Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers.
        
        Returns:
            dict: Headers
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def _request(self, method: str, endpoint: str, 
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            data (dict, optional): Request body
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers()
        
        if headers:
            request_headers.update(headers)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Return response data
            if response.content:
                return response.json()
            else:
                return {}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            
            # Try to extract error information from response
            error_info = {
                "error": str(e),
                "status_code": None
            }
            
            if hasattr(e, 'response') and e.response is not None:
                error_info["status_code"] = e.response.status_code
                
                try:
                    error_data = e.response.json()
                    error_info["error_data"] = error_data
                except:
                    error_info["error_data"] = e.response.text
            
            raise APIError(error_info)
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
        """
        return self._request('GET', endpoint, params=params, headers=headers)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint (str): API endpoint
            data (dict, optional): Request body
            params (dict, optional): Query parameters
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
        """
        return self._request('POST', endpoint, params=params, data=data, headers=headers)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None,
           params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint (str): API endpoint
            data (dict, optional): Request body
            params (dict, optional): Query parameters
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
        """
        return self._request('PUT', endpoint, params=params, data=data, headers=headers)
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
        """
        return self._request('DELETE', endpoint, params=params, headers=headers)


class APIError(Exception):
    """Exception raised for API errors."""
    
    def __init__(self, error_info: Dict[str, Any]):
        """
        Initialize the exception.
        
        Args:
            error_info (dict): Error information
        """
        self.error_info = error_info
        self.status_code = error_info.get("status_code")
        self.error_data = error_info.get("error_data")
        
        super().__init__(str(error_info))


class FHIRClient(APIClient):
    """Client for FHIR (Fast Healthcare Interoperability Resources) API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                client_id: Optional[str] = None, client_secret: Optional[str] = None,
                auth_url: Optional[str] = None):
        """
        Initialize the FHIR client.
        
        Args:
            base_url (str): Base URL for the FHIR API
            api_key (str, optional): API key for authentication
            client_id (str, optional): OAuth client ID
            client_secret (str, optional): OAuth client secret
            auth_url (str, optional): OAuth authorization URL
        """
        super().__init__(base_url, api_key)
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.access_token = None
        self.token_expiry = None
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with authentication.
        
        Returns:
            dict: Headers
        """
        headers = super()._get_headers()
        
        # Use access token if available
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        return headers
    
    def _authenticate(self) -> bool:
        """
        Authenticate with the FHIR server using OAuth.
        
        Returns:
            bool: True if authentication succeeded, False otherwise
        """
        if not self.client_id or not self.client_secret or not self.auth_url:
            return False
        
        try:
            response = requests.post(
                url=self.auth_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                },
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data.get('access_token')
            
            # Calculate token expiry time
            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            return bool(self.access_token)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def _ensure_authenticated(self):
        """
        Ensure the client is authenticated.
        
        Raises:
            APIError: If authentication fails
        """
        # Check if authentication is needed
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return
        
        # Try to authenticate
        if not self._authenticate():
            raise APIError({
                "error": "Authentication failed",
                "status_code": 401
            })
    
    def _request(self, method: str, endpoint: str, 
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make an API request with authentication.
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            data (dict, optional): Request body
            headers (dict, optional): Additional headers
            
        Returns:
            dict: Response data
        """
        # Ensure authentication if using OAuth
        if self.client_id and self.client_secret:
            self._ensure_authenticated()
        
        return super()._request(method, endpoint, params, data, headers)
    
    def search_patients(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for patients.
        
        Args:
            params (dict, optional): Search parameters
            
        Returns:
            dict: Search results
        """
        return self.get('Patient', params=params)
    
    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Get a patient by ID.
        
        Args:
            patient_id (str): Patient ID
            
        Returns:
            dict: Patient data
        """
        return self.get(f'Patient/{patient_id}')
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new patient.
        
        Args:
            patient_data (dict): Patient data
            
        Returns:
            dict: Created patient data
        """
        return self.post('Patient', data=patient_data)
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a patient.
        
        Args:
            patient_id (str): Patient ID
            patient_data (dict): Patient data
            
        Returns:
            dict: Updated patient data
        """
        return self.put(f'Patient/{patient_id}', data=patient_data)
    
    def search_observations(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for observations.
        
        Args:
            params (dict, optional): Search parameters
            
        Returns:
            dict: Search results
        """
        return self.get('Observation', params=params)
    
    def get_observation(self, observation_id: str) -> Dict[str, Any]:
        """
        Get an observation by ID.
        
        Args:
            observation_id (str): Observation ID
            
        Returns:
            dict: Observation data
        """
        return self.get(f'Observation/{observation_id}')
    
    def create_observation(self, observation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new observation.
        
        Args:
            observation_data (dict): Observation data
            
        Returns:
            dict: Created observation data
        """
        return self.post('Observation', data=observation_data)
    
    def search_conditions(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for conditions.
        
        Args:
            params (dict, optional): Search parameters
            
        Returns:
            dict: Search results
        """
        return self.get('Condition', params=params)
    
    def get_patient_everything(self, patient_id: str) -> Dict[str, Any]:
        """
        Get all resources for a patient.
        
        Args:
            patient_id (str): Patient ID
            
        Returns:
            dict: Patient resources
        """
        return self.get(f'Patient/{patient_id}/$everything')


class EHRIntegration:
    """Integration with Electronic Health Record systems."""
    
    def __init__(self, ehr_type: str, connection_params: Dict[str, Any]):
        """
        Initialize the EHR integration.
        
        Args:
            ehr_type (str): Type of EHR system
            connection_params (dict): Connection parameters
        """
        self.ehr_type = ehr_type
        self.connection_params = connection_params
        self.client = self._create_client()
        self.logger = logging.getLogger(__name__)
    
    def _create_client(self) -> Optional[APIClient]:
        """
        Create an API client for the EHR system.
        
        Returns:
            APIClient: API client
        """
        if self.ehr_type == "fhir":
            return FHIRClient(
                base_url=self.connection_params.get("base_url", ""),
                api_key=self.connection_params.get("api_key"),
                client_id=self.connection_params.get("client_id"),
                client_secret=self.connection_params.get("client_secret"),
                auth_url=self.connection_params.get("auth_url")
            )
        else:
            self.logger.error(f"Unsupported EHR type: {self.ehr_type}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test the connection to the EHR system.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            # Make a simple request to test the connection
            if isinstance(self.client, FHIRClient):
                # Try to access the metadata endpoint
                self.client.get('metadata')
            
            return True
        except APIError as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def import_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Import a patient from the EHR system.
        
        Args:
            patient_id (str): Patient ID in the EHR system
            
        Returns:
            dict: Imported patient data if successful, None otherwise
        """
        if not self.client:
            return None
        
        try:
            if isinstance(self.client, FHIRClient):
                # Get patient data
                patient_data = self.client.get_patient(patient_id)
                
                # Transform FHIR patient to internal format
                return self._transform_fhir_patient(patient_data)
            
            return None
        except APIError as e:
            self.logger.error(f"Failed to import patient: {e}")
            return None
    
    def import_patient_records(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Import all records for a patient from the EHR system.
        
        Args:
            patient_id (str): Patient ID in the EHR system
            
        Returns:
            dict: Imported patient records if successful, None otherwise
        """
        if not self.client:
            return None
        
        try:
            if isinstance(self.client, FHIRClient):
                # Get all patient resources
                patient_resources = self.client.get_patient_everything(patient_id)
                
                # Transform FHIR resources to internal format
                return self._transform_fhir_resources(patient_resources)
            
            return None
        except APIError as e:
            self.logger.error(f"Failed to import patient records: {e}")
            return None
    
    def export_assessment(self, assessment_data: Dict[str, Any]) -> Optional[str]:
        """
        Export an assessment to the EHR system.
        
        Args:
            assessment_data (dict): Assessment data
            
        Returns:
            str: Exported resource ID if successful, None otherwise
        """
        if not self.client:
            return None
        
        try:
            if isinstance(self.client, FHIRClient):
                # Transform assessment to FHIR observation
                observation_data = self._transform_assessment_to_fhir(assessment_data)
                
                # Create observation in FHIR
                response = self.client.create_observation(observation_data)
                
                # Return the resource ID
                return response.get("id")
            
            return None
        except APIError as e:
            self.logger.error(f"Failed to export assessment: {e}")
            return None
    
    def _transform_fhir_patient(self, fhir_patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a FHIR patient resource to internal format.
        
        Args:
            fhir_patient (dict): FHIR patient resource
            
        Returns:
            dict: Internal patient format
        """
        # Extract basic patient information
        patient_id = fhir_patient.get("id", "")
        
        # Get name
        name = ""
        if "name" in fhir_patient and fhir_patient["name"]:
            name_obj = fhir_patient["name"][0]
            given = name_obj.get("given", [])
            family = name_obj.get("family", "")
            
            if given:
                name = " ".join(given)
            
            if family:
                if name:
                    name += " "
                name += family
        
        # Get gender
        gender = fhir_patient.get("gender", "")
        
        # Get birth date
        birth_date = fhir_patient.get("birthDate", "")
        
        # Calculate age if birth date is available
        age = None
        if birth_date:
            try:
                birth_date_obj = datetime.strptime(birth_date, "%Y-%m-%d")
                age = (datetime.now() - birth_date_obj).days // 365
            except ValueError:
                pass
        
        # Transform to internal format
        return {
            "patient_id": f"FHIR-{patient_id}",
            "name": name,
            "gender": gender,
            "birth_date": birth_date,
            "age": age,
            "source": "fhir",
            "source_id": patient_id,
            "raw_data": fhir_patient
        }
    
    def _transform_fhir_resources(self, fhir_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform FHIR resources to internal format.
        
        Args:
            fhir_resources (dict): FHIR resources
            
        Returns:
            dict: Internal format
        """
        result = {
            "patient": None,
            "observations": [],
            "conditions": [],
            "medications": [],
            "procedures": []
        }
        
        # Extract resources from bundle
        if "entry" in fhir_resources:
            for entry in fhir_resources["entry"]:
                if "resource" not in entry:
                    continue
                
                resource = entry["resource"]
                resource_type = resource.get("resourceType")
                
                if resource_type == "Patient":
                    result["patient"] = self._transform_fhir_patient(resource)
                elif resource_type == "Observation":
                    # Transform observation
                    observation = self._transform_fhir_observation(resource)
                    if observation:
                        result["observations"].append(observation)
                elif resource_type == "Condition":
                    # Transform condition
                    condition = self._transform_fhir_condition(resource)
                    if condition:
                        result["conditions"].append(condition)
                elif resource_type == "MedicationRequest":
                    # Transform medication
                    medication = self._transform_fhir_medication(resource)
                    if medication:
                        result["medications"].append(medication)
                elif resource_type == "Procedure":
                    # Transform procedure
                    procedure = self._transform_fhir_procedure(resource)
                    if procedure:
                        result["procedures"].append(procedure)
        
        return result
    
    def _transform_fhir_observation(self, fhir_observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a FHIR observation resource to internal format.
        
        Args:
            fhir_observation (dict): FHIR observation resource
            
        Returns:
            dict: Internal observation format
        """
        observation_id = fhir_observation.get("id", "")
        
        # Get code
        code = ""
        code_display = ""
        if "code" in fhir_observation and "coding" in fhir_observation["code"]:
            coding = fhir_observation["code"]["coding"]
            if coding:
                code = coding[0].get("code", "")
                code_display = coding[0].get("display", "")
        
        # Get value
        value = None
        value_type = None
        
        if "valueQuantity" in fhir_observation:
            value = fhir_observation["valueQuantity"].get("value")
            value_type = "quantity"
        elif "valueString" in fhir_observation:
            value = fhir_observation["valueString"]
            value_type = "string"
        elif "valueBoolean" in fhir_observation:
            value = fhir_observation["valueBoolean"]
            value_type = "boolean"
        elif "valueInteger" in fhir_observation:
            value = fhir_observation["valueInteger"]
            value_type = "integer"
        elif "valueCodeableConcept" in fhir_observation:
            value_cc = fhir_observation["valueCodeableConcept"]
            if "coding" in value_cc and value_cc["coding"]:
                value = value_cc["coding"][0].get("code")
                value_type = "code"
        
        # Get date
        effective_date = None
        if "effectiveDateTime" in fhir_observation:
            effective_date = fhir_observation["effectiveDateTime"]
        
        # Transform to internal format
        return {
            "observation_id": observation_id,
            "code": code,
            "display": code_display,
            "value": value,
            "value_type": value_type,
            "date": effective_date,
            "source": "fhir",
            "source_id": observation_id,
            "raw_data": fhir_observation
        }
    
    def _transform_fhir_condition(self, fhir_condition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a FHIR condition resource to internal format.
        
        Args:
            fhir_condition (dict): FHIR condition resource
            
        Returns:
            dict: Internal condition format
        """
        condition_id = fhir_condition.get("id", "")
        
        # Get code
        code = ""
        code_display = ""
        if "code" in fhir_condition and "coding" in fhir_condition["code"]:
            coding = fhir_condition["code"]["coding"]
            if coding:
                code = coding[0].get("code", "")
                code_display = coding[0].get("display", "")
        
        # Get status
        status = fhir_condition.get("clinicalStatus", {})
        if "coding" in status and status["coding"]:
            status = status["coding"][0].get("code", "")
        else:
            status = ""
        
        # Get onset date
        onset_date = None
        if "onsetDateTime" in fhir_condition:
            onset_date = fhir_condition["onsetDateTime"]
        
        # Transform to internal format
        return {
            "condition_id": condition_id,
            "code": code,
            "display": code_display,
            "status": status,
            "onset_date": onset_date,
            "source": "fhir",
            "source_id": condition_id,
            "raw_data": fhir_condition
        }
    
    def _transform_fhir_medication(self, fhir_medication: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a FHIR medication resource to internal format.
        
        Args:
            fhir_medication (dict): FHIR medication resource
            
        Returns:
            dict: Internal medication format
        """
        medication_id = fhir_medication.get("id", "")
        
        # Get medication code
        code = ""
        code_display = ""
        if "medicationCodeableConcept" in fhir_medication:
            med_code = fhir_medication["medicationCodeableConcept"]
            if "coding" in med_code and med_code["coding"]:
                code = med_code["coding"][0].get("code", "")
                code_display = med_code["coding"][0].get("display", "")
        
        # Get status
        status = fhir_medication.get("status", "")
        
        # Get date
        authored_date = None
        if "authoredOn" in fhir_medication:
            authored_date = fhir_medication["authoredOn"]
        
        # Transform to internal format
        return {
            "medication_id": medication_id,
            "code": code,
            "display": code_display,
            "status": status,
            "date": authored_date,
            "source": "fhir",
            "source_id": medication_id,
            "raw_data": fhir_medication
        }
    
    def _transform_fhir_procedure(self, fhir_procedure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a FHIR procedure resource to internal format.
        
        Args:
            fhir_procedure (dict): FHIR procedure resource
            
        Returns:
            dict: Internal procedure format
        """
        procedure_id = fhir_procedure.get("id", "")
        
        # Get code
        code = ""
        code_display = ""
        if "code" in fhir_procedure and "coding" in fhir_procedure["code"]:
            coding = fhir_procedure["code"]["coding"]
            if coding:
                code = coding[0].get("code", "")
                code_display = coding[0].get("display", "")
        
        # Get status
        status = fhir_procedure.get("status", "")
        
        # Get date
        performed_date = None
        if "performedDateTime" in fhir_procedure:
            performed_date = fhir_procedure["performedDateTime"]
        
        # Transform to internal format
        return {
            "procedure_id": procedure_id,
            "code": code,
            "display": code_display,
            "status": status,
            "date": performed_date,
            "source": "fhir",
            "source_id": procedure_id,
            "raw_data": fhir_procedure
        }
    
    def _transform_assessment_to_fhir(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform an assessment to a FHIR observation.
        
        Args:
            assessment_data (dict): Assessment data
            
        Returns:
            dict: FHIR observation
        """
        # Create a FHIR Observation resource
        observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey"
                        }
                    ],
                    "text": "Mental Health Assessment"
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "89204-2",
                        "display": "Mental health Assessment note"
                    }
                ],
                "text": "MH-Net Mental Health Assessment"
            },
            "subject": {
                "reference": f"Patient/{assessment_data.get('patient_id', '').replace('FHIR-', '')}"
            },
            "effectiveDateTime": assessment_data.get("timestamp", datetime.now().isoformat()),
            "issued": datetime.now().isoformat(),
            "performer": [
                {
                    "display": assessment_data.get("metadata", {}).get("clinician", "MH-Net System")
                }
            ],
            "component": []
        }
        
        # Add risk scores as components
        risk_scores = assessment_data.get("risk_scores", {})
        for condition, score in risk_scores.items():
            component = {
                "code": {
                    "coding": [
                        {
                            "system": "http://mhnet.org/codes",
                            "code": f"risk-{condition.lower()}",
                            "display": f"{condition} Risk Score"
                        }
                    ],
                    "text": f"{condition} Risk Score"
                },
                "valueQuantity": {
                    "value": score,
                    "unit": "score",
                    "system": "http://unitsofmeasure.org",
                    "code": "score"
                }
            }
            
            observation["component"].append(component)
        
        # Add text input if available
        text_input = assessment_data.get("text_input")
        if text_input:
            component = {
                "code": {
                    "coding": [
                        {
                            "system": "http://mhnet.org/codes",
                            "code": "text-input",
                            "display": "Text Input"
                        }
                    ],
                    "text": "Patient Text Input"
                },
                "valueString": text_input
            }
            
            observation["component"].append(component)
        
        return observation


class ExternalAPIManager:
    """Manager for external API integrations."""
    
    def __init__(self):
        """Initialize the ExternalAPIManager."""
        self.api_clients = {}
        self.ehr_integrations = {}
        self.logger = logging.getLogger(__name__)
    
    def register_api_client(self, name: str, client: APIClient) -> bool:
        """
        Register an API client.
        
        Args:
            name (str): Client name
            client (APIClient): API client
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name in self.api_clients:
            self.logger.warning(f"API client '{name}' already registered. Replacing.")
        
        self.api_clients[name] = client
        return True
    
    def get_api_client(self, name: str) -> Optional[APIClient]:
        """
        Get an API client by name.
        
        Args:
            name (str): Client name
            
        Returns:
            APIClient: API client if found, None otherwise
        """
        return self.api_clients.get(name)
    
    def register_ehr_integration(self, name: str, integration: EHRIntegration) -> bool:
        """
        Register an EHR integration.
        
        Args:
            name (str): Integration name
            integration (EHRIntegration): EHR integration
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name in self.ehr_integrations:
            self.logger.warning(f"EHR integration '{name}' already registered. Replacing.")
        
        self.ehr_integrations[name] = integration
        return True
    
    def get_ehr_integration(self, name: str) -> Optional[EHRIntegration]:
        """
        Get an EHR integration by name.
        
        Args:
            name (str): Integration name
            
        Returns:
            EHRIntegration: EHR integration if found, None otherwise
        """
        return self.ehr_integrations.get(name)
    
    def create_fhir_integration(self, name: str, 
                              base_url: str,
                              api_key: Optional[str] = None,
                              client_id: Optional[str] = None,
                              client_secret: Optional[str] = None,
                              auth_url: Optional[str] = None) -> bool:
        """
        Create and register a FHIR integration.
        
        Args:
            name (str): Integration name
            base_url (str): Base URL for the FHIR API
            api_key (str, optional): API key for authentication
            client_id (str, optional): OAuth client ID
            client_secret (str, optional): OAuth client secret
            auth_url (str, optional): OAuth authorization URL
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create connection parameters
            connection_params = {
                "base_url": base_url,
                "api_key": api_key,
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_url": auth_url
            }
            
            # Create integration
            integration = EHRIntegration("fhir", connection_params)
            
            # Test connection
            if not integration.test_connection():
                self.logger.error(f"Failed to connect to FHIR server at {base_url}")
                return False
            
            # Register integration
            return self.register_ehr_integration(name, integration)
        except Exception as e:
            self.logger.error(f"Failed to create FHIR integration: {e}")
            return False


# Singleton instance
api_manager = ExternalAPIManager()
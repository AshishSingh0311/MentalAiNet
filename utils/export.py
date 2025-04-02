"""
Export and reporting utilities for MH-Net.

This module provides functionality for exporting assessments, reports, 
and model results in various formats including PDF, CSV, and HTML.
"""

import os
import base64
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from json2html import json2html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PDF(FPDF):
    """Extended FPDF class with custom header and footer."""
    
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        # Set font
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'MH-Net Assessment Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Set font
        self.set_font('Arial', 'I', 8)
        # Add page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        # Add timestamp
        self.cell(-30, 10, datetime.now().strftime("%Y-%m-%d %H:%M"), 0, 0, 'R')


def create_assessment_pdf(assessment_data, output_path=None):
    """
    Create a PDF report for a patient assessment.
    
    Args:
        assessment_data (dict): Dictionary containing assessment data
        output_path (str, optional): Path to save the PDF file. If None, returns the PDF as bytes.
        
    Returns:
        bytes or None: If output_path is None, returns the PDF as bytes, otherwise None
    """
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Set font
    pdf.set_font('Arial', '', 12)
    
    # Patient Information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Check that metadata exists
    if "metadata" in assessment_data:
        metadata = assessment_data["metadata"]
        pdf.cell(0, 10, f"Patient ID: {assessment_data.get('patient_id', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Age: {metadata.get('age', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Gender: {metadata.get('gender', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Assessment Date: {metadata.get('assessment_date', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Clinician: {metadata.get('clinician', 'N/A')}", 0, 1)
        pdf.cell(0, 10, f"Session Type: {metadata.get('session_type', 'N/A')}", 0, 1)
    else:
        pdf.cell(0, 10, f"Patient ID: {assessment_data.get('patient_id', 'N/A')}", 0, 1)
        pdf.cell(0, 10, "No additional metadata available", 0, 1)
    
    pdf.ln(5)
    
    # Assessment Results
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Assessment Results', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Risk Scores
    if "risk_scores" in assessment_data:
        risk_scores = assessment_data["risk_scores"]
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Risk Scores:', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Create a table for risk scores
        for condition, score in risk_scores.items():
            # Determine risk level text and color
            if score >= 0.7:
                risk_level = "High"
                pdf.set_text_color(255, 0, 0)  # Red
            elif score >= 0.4:
                risk_level = "Moderate"
                pdf.set_text_color(255, 165, 0)  # Orange
            else:
                risk_level = "Low"
                pdf.set_text_color(0, 128, 0)  # Green
                
            pdf.cell(60, 10, condition, 1)
            pdf.cell(40, 10, f"{score:.2f}", 1)
            pdf.cell(40, 10, risk_level, 1)
            pdf.ln()
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.cell(0, 10, "No risk scores available", 0, 1)
    
    pdf.ln(5)
    
    # Patient Input
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Patient Input', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Text input if available
    if assessment_data.get("text_input"):
        pdf.multi_cell(0, 10, f"Text Input: {assessment_data['text_input']}")
    else:
        pdf.cell(0, 10, "No text input available", 0, 1)
    
    # Input modalities
    modalities = []
    if assessment_data.get("has_audio"):
        modalities.append("Audio")
    if assessment_data.get("has_physio"):
        modalities.append("Physiological")
    if assessment_data.get("has_imaging"):
        modalities.append("Imaging")
    
    if modalities:
        pdf.cell(0, 10, f"Additional data modalities: {', '.join(modalities)}", 0, 1)
    
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Clinical Recommendations', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Find primary risk factor
    if "risk_scores" in assessment_data:
        primary_condition = max(assessment_data["risk_scores"].items(), key=lambda x: x[1])
        condition = primary_condition[0]
        score = primary_condition[1]
        
        # Generate recommendations based on condition and score
        recommendations = get_recommendations(condition, score)
        
        for i, rec in enumerate(recommendations, 1):
            pdf.multi_cell(0, 10, f"{i}. {rec}")
    else:
        pdf.cell(0, 10, "No recommendations available without risk scores", 0, 1)
    
    # Final notes
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 10, "Note: This report is generated by the MH-Net system and should be reviewed by a qualified mental health professional. The system provides decision support but does not replace clinical judgment.")
    
    # Output PDF
    if output_path:
        pdf.output(output_path)
        return None
    else:
        # Return PDF as bytes
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue()


def get_recommendations(condition, risk_score):
    """
    Generate clinical recommendations based on condition and risk score.
    
    Args:
        condition (str): The primary condition
        risk_score (float): The risk score (0-1)
        
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    # General recommendation for all conditions
    recommendations.append("Regular follow-up appointments to monitor progress and adjust treatment as needed.")
    
    # Risk level specific recommendations
    if risk_score >= 0.7:
        recommendations.append(f"High risk of {condition} detected. Consider immediate clinical intervention.")
        recommendations.append("Comprehensive clinical assessment recommended within 24-48 hours.")
    elif risk_score >= 0.4:
        recommendations.append(f"Moderate risk of {condition} detected. Clinical monitoring recommended.")
        recommendations.append("Follow-up assessment within 1-2 weeks to track symptoms.")
    else:
        recommendations.append(f"Low risk of {condition} detected. Continued monitoring recommended.")
        recommendations.append("Routine follow-up within 4-6 weeks.")
    
    # Condition-specific recommendations
    if condition == "Depression":
        recommendations.append("Consider evidence-based psychological interventions such as CBT or IPT.")
        if risk_score >= 0.6:
            recommendations.append("Evaluate for antidepressant medication in conjunction with therapy.")
        recommendations.append("Screen for suicidal ideation and establish safety plan if indicated.")
        
    elif condition == "Anxiety":
        recommendations.append("Consider evidence-based psychological interventions such as CBT or mindfulness-based approaches.")
        recommendations.append("Psychoeducation about anxiety management strategies and relaxation techniques.")
        if risk_score >= 0.6:
            recommendations.append("Consider anxiolytic medication for short-term management if symptoms are severe.")
        
    elif condition == "PTSD":
        recommendations.append("Consider trauma-focused psychological interventions such as CPT or EMDR.")
        recommendations.append("Evaluate for comorbid conditions including depression and substance use.")
        recommendations.append("Provide psychoeducation on trauma responses and coping strategies.")
        
    elif condition == "Bipolar":
        recommendations.append("Comprehensive mood monitoring recommended.")
        recommendations.append("Evaluate for mood stabilization medication.")
        recommendations.append("Psychoeducation about bipolar disorder and importance of regular sleep patterns.")
        if risk_score >= 0.6:
            recommendations.append("Assess for manic/hypomanic symptoms and develop crisis management plan.")
        
    elif condition == "Schizophrenia":
        recommendations.append("Comprehensive psychiatric evaluation recommended.")
        recommendations.append("Evaluate for antipsychotic medication and monitor efficacy/side effects.")
        recommendations.append("Psychosocial support and family education recommended.")
        if risk_score >= 0.6:
            recommendations.append("Assess for positive symptoms and develop safety/crisis management plan.")
    
    return recommendations


def export_to_csv(data, output_path):
    """
    Export data to CSV format.
    
    Args:
        data (dict/list): Data to export
        output_path (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert data to DataFrame
        if isinstance(data, dict):
            # Handle nested dictionaries
            flattened_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        flattened_data[f"{key}_{inner_key}"] = inner_value
                else:
                    flattened_data[key] = value
            
            df = pd.DataFrame([flattened_data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return False
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        return False


def export_to_html(data, output_path=None):
    """
    Export data to HTML format.
    
    Args:
        data (dict/list): Data to export
        output_path (str, optional): Path to save the HTML file. If None, returns HTML as string.
        
    Returns:
        str or bool: HTML string if output_path is None, otherwise True if successful, False otherwise
    """
    try:
        # Convert data to HTML
        html_content = json2html.convert(json=data)
        
        # Add basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MH-Net Export</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1, h2 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MH-Net Export</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                {html_content}
            </div>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(styled_html)
            return True
        else:
            return styled_html
    except Exception as e:
        print(f"Error exporting to HTML: {str(e)}")
        return False


def export_model_summary(model_data, output_path=None):
    """
    Export model summary to various formats.
    
    Args:
        model_data (dict): Dictionary containing model information
        output_path (str, optional): Path to save the model summary. If None, returns summary as dict.
        
    Returns:
        dict or bool: Model summary as dict if output_path is None, otherwise True if successful, False otherwise
    """
    try:
        # Create model summary
        summary = {
            "model_type": model_data.get("type", "Unknown"),
            "architecture": {
                "embed_dim": model_data.get("embed_dim", None),
                "num_heads": model_data.get("num_heads", None),
                "num_layers": model_data.get("num_layers", None),
                "dropout_rate": model_data.get("dropout_rate", None),
                "hidden_dim": model_data.get("hidden_dim", None),
                "activation": model_data.get("activation", None),
            },
            "training_time": model_data.get("training_time", "Unknown"),
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "performance_metrics": {
                "accuracy": model_data.get("accuracy", None),
                "precision": model_data.get("precision", None),
                "recall": model_data.get("recall", None),
                "f1_score": model_data.get("f1_score", None),
                "roc_auc": model_data.get("roc_auc", None),
            },
            "parameters": model_data.get("parameters", "Unknown"),
        }
        
        if output_path:
            # Determine format from file extension
            ext = os.path.splitext(output_path)[1].lower()
            
            if ext == '.json':
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=4)
            elif ext == '.csv':
                export_to_csv(summary, output_path)
            elif ext == '.html':
                export_to_html(summary, output_path)
            elif ext == '.pdf':
                # Create a PDF for model summary
                pdf = PDF()
                pdf.alias_nb_pages()
                pdf.add_page()
                
                # Model Information
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, 'Model Summary Report', 0, 1)
                
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Model Information', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                pdf.cell(0, 10, f"Model Type: {summary['model_type']}", 0, 1)
                pdf.cell(0, 10, f"Training Date: {summary['trained_date']}", 0, 1)
                
                # Architecture
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Model Architecture', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                arch = summary['architecture']
                for key, value in arch.items():
                    if value is not None:
                        pdf.cell(0, 10, f"{key}: {value}", 0, 1)
                
                # Performance Metrics
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Performance Metrics', 0, 1)
                pdf.set_font('Arial', '', 12)
                
                metrics = summary['performance_metrics']
                for key, value in metrics.items():
                    if value is not None:
                        pdf.cell(0, 10, f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}", 0, 1)
                
                pdf.output(output_path)
            else:
                # Unsupported format
                return False
            
            return True
        else:
            return summary
    except Exception as e:
        print(f"Error exporting model summary: {str(e)}")
        return False


def plot_to_image(fig):
    """
    Convert a matplotlib figure to a PNG image.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        bytes: The PNG image as bytes
    """
    # Save figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def get_downloadable_link(data, filename, mime_type):
    """
    Generate a downloadable link for data.
    
    Args:
        data (bytes): The data to download
        filename (str): The filename for the download
        mime_type (str): The MIME type of the data
        
    Returns:
        str: HTML for a downloadable link
    """
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
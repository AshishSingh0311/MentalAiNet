"""
Clinical recommendation module for MH-Net.

This module provides functionality for generating clinical recommendations
based on assessment results and best practices.
"""

import json
import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple


class RecommendationEngine:
    """Engine for generating clinical recommendations."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            knowledge_base_path (str, optional): Path to knowledge base file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
    
    def _load_knowledge_base(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the knowledge base.
        
        Args:
            path (str, optional): Path to knowledge base file
            
        Returns:
            dict: Knowledge base data
        """
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Return default knowledge base if file is invalid
                return self._get_default_knowledge_base()
        
        # Return default knowledge base if no path provided
        return self._get_default_knowledge_base()
    
    def _get_default_knowledge_base(self) -> Dict[str, Any]:
        """
        Get the default knowledge base.
        
        Returns:
            dict: Default knowledge base data
        """
        return {
            "conditions": {
                "Depression": {
                    "interventions": {
                        "high_risk": [
                            {
                                "type": "clinical",
                                "name": "Immediate psychiatric evaluation",
                                "description": "Schedule an immediate psychiatric evaluation to assess suicide risk and potential need for hospitalization.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Antidepressant medication",
                                "description": "Consider starting an appropriate antidepressant (SSRI or SNRI) under close monitoring for adverse effects and clinical response.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Cognitive Behavioral Therapy (CBT)",
                                "description": "Initiate intensive CBT with a focus on behavioral activation and cognitive restructuring.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Crisis management plan",
                                "description": "Develop a detailed crisis management plan with the patient and their support network.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "monitoring",
                                "name": "Frequent follow-up",
                                "description": "Schedule frequent follow-up appointments (weekly or biweekly) to monitor symptoms and treatment response.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "moderate_risk": [
                            {
                                "type": "clinical",
                                "name": "Comprehensive evaluation",
                                "description": "Conduct a comprehensive evaluation of depressive symptoms, functional impairment, and treatment history.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Cognitive Behavioral Therapy (CBT)",
                                "description": "Refer for CBT focusing on negative thought patterns and behavioral activation.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Interpersonal Therapy (IPT)",
                                "description": "Consider IPT to address interpersonal issues that may be contributing to depression.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Medication evaluation",
                                "description": "Evaluate the need for antidepressant medication based on symptom severity and functional impairment.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Exercise program",
                                "description": "Recommend a regular exercise program, ideally 30 minutes of moderate activity at least 3 times per week.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            }
                        ],
                        "low_risk": [
                            {
                                "type": "monitoring",
                                "name": "Symptom monitoring",
                                "description": "Implement regular monitoring of depressive symptoms using standardized measures.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "education",
                                "name": "Psychoeducation",
                                "description": "Provide education about depression, including self-management strategies and warning signs.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Sleep hygiene",
                                "description": "Promote healthy sleep habits and regular sleep schedule.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Physical activity",
                                "description": "Encourage regular physical activity, starting with small, achievable goals.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Social support",
                                "description": "Help identify and strengthen social support networks.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ]
                    },
                    "assessment_tools": [
                        {
                            "name": "PHQ-9",
                            "description": "Patient Health Questionnaire-9",
                            "cutoffs": {
                                "minimal": [0, 4],
                                "mild": [5, 9],
                                "moderate": [10, 14],
                                "moderately_severe": [15, 19],
                                "severe": [20, 27]
                            }
                        },
                        {
                            "name": "BDI-II",
                            "description": "Beck Depression Inventory-II",
                            "cutoffs": {
                                "minimal": [0, 13],
                                "mild": [14, 19],
                                "moderate": [20, 28],
                                "severe": [29, 63]
                            }
                        }
                    ],
                    "comorbidities": [
                        {
                            "condition": "Anxiety",
                            "adjustment": "Consider transdiagnostic approaches such as Unified Protocol or ACT if comorbid with anxiety."
                        },
                        {
                            "condition": "Substance Use",
                            "adjustment": "Address substance use concurrently; consider integrated treatment approaches."
                        },
                        {
                            "condition": "PTSD",
                            "adjustment": "Trauma-focused therapy may need to precede or be integrated with depression treatment."
                        }
                    ]
                },
                "Anxiety": {
                    "interventions": {
                        "high_risk": [
                            {
                                "type": "clinical",
                                "name": "Psychiatric evaluation",
                                "description": "Comprehensive psychiatric evaluation to assess anxiety severity, comorbidities, and treatment options.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Cognitive Behavioral Therapy (CBT)",
                                "description": "Intensive CBT with emphasis on exposure techniques and cognitive restructuring.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Medication management",
                                "description": "Consider starting SSRI/SNRI for long-term management and possibly short-term benzodiazepines if severe symptoms (with caution).",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Crisis management skills",
                                "description": "Teach specific skills for managing acute anxiety and panic episodes.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Stress reduction",
                                "description": "Implement stress reduction strategies including mindfulness and relaxation techniques.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ],
                        "moderate_risk": [
                            {
                                "type": "therapy",
                                "name": "Cognitive Behavioral Therapy (CBT)",
                                "description": "Standard CBT protocol focusing on anxiety management techniques and exposure.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Acceptance and Commitment Therapy (ACT)",
                                "description": "ACT can help develop psychological flexibility and mindfulness skills.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "medication",
                                "name": "Medication consideration",
                                "description": "Evaluate appropriateness of medication (typically SSRIs or SNRIs) based on symptom severity and patient preference.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "skills",
                                "name": "Anxiety management skills",
                                "description": "Teach specific skills including relaxation training, deep breathing, and mindfulness.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Regular exercise",
                                "description": "Implement regular aerobic exercise program (30 minutes, 3-5 times weekly).",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            }
                        ],
                        "low_risk": [
                            {
                                "type": "education",
                                "name": "Psychoeducation",
                                "description": "Provide education about anxiety, including nature of symptoms, triggers, and self-management.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Basic anxiety management",
                                "description": "Teach foundational anxiety management skills like deep breathing and progressive muscle relaxation.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Lifestyle modifications",
                                "description": "Recommend reducing caffeine, improving sleep hygiene, and regular physical activity.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "therapy",
                                "name": "Self-guided resources",
                                "description": "Recommend evidence-based self-help books or digital mental health applications for anxiety.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "monitoring",
                                "name": "Regular monitoring",
                                "description": "Implement periodic monitoring of anxiety symptoms using standardized measures.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ]
                    },
                    "assessment_tools": [
                        {
                            "name": "GAD-7",
                            "description": "Generalized Anxiety Disorder 7-item scale",
                            "cutoffs": {
                                "minimal": [0, 4],
                                "mild": [5, 9],
                                "moderate": [10, 14],
                                "severe": [15, 21]
                            }
                        },
                        {
                            "name": "BAI",
                            "description": "Beck Anxiety Inventory",
                            "cutoffs": {
                                "minimal": [0, 7],
                                "mild": [8, 15],
                                "moderate": [16, 25],
                                "severe": [26, 63]
                            }
                        }
                    ],
                    "comorbidities": [
                        {
                            "condition": "Depression",
                            "adjustment": "Address both conditions; SSRIs may be particularly beneficial."
                        },
                        {
                            "condition": "Substance Use",
                            "adjustment": "Address substance use that may be exacerbating anxiety."
                        },
                        {
                            "condition": "Insomnia",
                            "adjustment": "Incorporate sleep interventions as part of treatment plan."
                        }
                    ]
                },
                "PTSD": {
                    "interventions": {
                        "high_risk": [
                            {
                                "type": "clinical",
                                "name": "Specialized trauma evaluation",
                                "description": "Comprehensive assessment by a trauma specialist to determine severity and treatment needs.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Trauma-focused psychotherapy",
                                "description": "Evidence-based trauma therapies such as Prolonged Exposure (PE) or Cognitive Processing Therapy (CPT).",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Medication management",
                                "description": "Consider SSRIs/SNRIs as first-line pharmacotherapy; prazosin may help with nightmares.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Stabilization skills",
                                "description": "Teach grounding techniques, emotion regulation, and distress tolerance skills.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Safety planning",
                                "description": "Develop comprehensive safety plan if there are suicidal thoughts or high-risk behaviors.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "moderate_risk": [
                            {
                                "type": "therapy",
                                "name": "Trauma-focused therapy",
                                "description": "Evidence-based trauma treatment such as EMDR, PE, or CPT.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Emotion regulation skills",
                                "description": "Provide training in identifying, labeling, and managing intense emotions.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Medication consideration",
                                "description": "Evaluate need for medication treatment, particularly SSRIs or SNRIs.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Sleep improvement",
                                "description": "Address sleep disturbances through sleep hygiene and specific interventions for nightmares.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "support",
                                "name": "Support system",
                                "description": "Strengthen social support network and consider trauma-informed family education.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ],
                        "low_risk": [
                            {
                                "type": "education",
                                "name": "Trauma psychoeducation",
                                "description": "Provide education about PTSD, trauma responses, and recovery process.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Basic coping skills",
                                "description": "Teach grounding techniques, mindfulness, and stress management strategies.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Self-care routine",
                                "description": "Encourage establishment of healthy routines including exercise, sleep, and nutrition.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "support",
                                "name": "Peer support",
                                "description": "Consider connection to peer support resources or trauma support groups.",
                                "evidence_level": "C",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "monitoring",
                                "name": "Regular assessment",
                                "description": "Monitor symptoms for any worsening that would require more intensive intervention.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ]
                    },
                    "assessment_tools": [
                        {
                            "name": "PCL-5",
                            "description": "PTSD Checklist for DSM-5",
                            "cutoffs": {
                                "minimal": [0, 32],
                                "probable PTSD": [33, 80]
                            }
                        },
                        {
                            "name": "CAPS-5",
                            "description": "Clinician-Administered PTSD Scale for DSM-5",
                            "cutoffs": {
                                "subclinical": [0, 19],
                                "mild": [20, 39],
                                "moderate": [40, 59],
                                "severe": [60, 80]
                            }
                        }
                    ],
                    "comorbidities": [
                        {
                            "condition": "Depression",
                            "adjustment": "Monitor for suicidal ideation; consider integrated treatment approach."
                        },
                        {
                            "condition": "Substance Use",
                            "adjustment": "Substance use as self-medication is common; address in parallel."
                        },
                        {
                            "condition": "Dissociation",
                            "adjustment": "May need stabilization work before trauma processing begins."
                        }
                    ]
                },
                "Bipolar": {
                    "interventions": {
                        "high_risk": [
                            {
                                "type": "clinical",
                                "name": "Immediate psychiatric evaluation",
                                "description": "Urgent psychiatric assessment for medication management and potential hospitalization if severe mania/hypomania or suicidality.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Mood stabilizer therapy",
                                "description": "Initiate or adjust mood stabilizers (lithium, valproate, carbamazepine) or atypical antipsychotics.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "monitoring",
                                "name": "Close monitoring",
                                "description": "Implement frequent monitoring of mood, medication effects, and safety concerns.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Bipolar-specific therapy",
                                "description": "Specialized therapy approaches such as Interpersonal and Social Rhythm Therapy (IPSRT) or psychoeducation.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Crisis management plan",
                                "description": "Develop detailed plan for recognizing warning signs and responding to mood episodes.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "moderate_risk": [
                            {
                                "type": "medication",
                                "name": "Medication optimization",
                                "description": "Optimize medication regimen; ensure adequate mood stabilization while minimizing side effects.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Specialized bipolar therapy",
                                "description": "Implement evidence-based approaches like IPSRT, CBT for bipolar disorder, or Family-Focused Therapy.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "education",
                                "name": "Comprehensive psychoeducation",
                                "description": "Provide detailed education about bipolar disorder, treatment, and relapse prevention.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Routine regulation",
                                "description": "Establish regular sleep, meal, and activity schedules to stabilize circadian rhythms.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "monitoring",
                                "name": "Mood tracking",
                                "description": "Implement daily mood monitoring to identify early warning signs of mood episodes.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "low_risk": [
                            {
                                "type": "education",
                                "name": "Basic psychoeducation",
                                "description": "Provide education about bipolar disorder, triggers, and importance of medication adherence.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Medication maintenance",
                                "description": "Regular medication review and adjustment as needed, with emphasis on long-term adherence.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Sleep regulation",
                                "description": "Establish consistent sleep-wake cycle with adequate sleep duration.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Support network",
                                "description": "Strengthen social support network and consider family education.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "monitoring",
                                "name": "Regular follow-up",
                                "description": "Schedule regular check-ins for early identification of mood changes.",
                                "evidence_level": "B",
                                "recommendation_strength": "Strong"
                            }
                        ]
                    },
                    "assessment_tools": [
                        {
                            "name": "YMRS",
                            "description": "Young Mania Rating Scale",
                            "cutoffs": {
                                "no mania": [0, 12],
                                "hypomania": [13, 19],
                                "mania": [20, 60]
                            }
                        },
                        {
                            "name": "MDQ",
                            "description": "Mood Disorder Questionnaire",
                            "cutoffs": {
                                "negative": [0, 6],
                                "positive": [7, 13]
                            }
                        },
                        {
                            "name": "QIDS",
                            "description": "Quick Inventory of Depressive Symptomatology",
                            "cutoffs": {
                                "normal": [0, 5],
                                "mild": [6, 10],
                                "moderate": [11, 15],
                                "severe": [16, 20],
                                "very severe": [21, 27]
                            }
                        }
                    ],
                    "comorbidities": [
                        {
                            "condition": "Anxiety",
                            "adjustment": "Careful medication selection to avoid exacerbating bipolar symptoms."
                        },
                        {
                            "condition": "Substance Use",
                            "adjustment": "Integrated treatment essential; substance use significantly worsens course."
                        },
                        {
                            "condition": "ADHD",
                            "adjustment": "Careful stimulant use if needed; prefer non-stimulant options when possible."
                        }
                    ]
                },
                "Schizophrenia": {
                    "interventions": {
                        "high_risk": [
                            {
                                "type": "clinical",
                                "name": "Comprehensive psychiatric evaluation",
                                "description": "Immediate psychiatric assessment to evaluate psychotic symptoms, safety risks, and need for hospitalization.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "medication",
                                "name": "Antipsychotic management",
                                "description": "Initiate or optimize antipsychotic medication, considering both first and second generation options.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Care coordination",
                                "description": "Implement intensive case management or Assertive Community Treatment (ACT) if available.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Family support and education",
                                "description": "Engage family in treatment planning and provide psychoeducation about schizophrenia.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "monitoring",
                                "name": "Close symptom monitoring",
                                "description": "Regular assessment of psychotic symptoms, medication side effects, and treatment adherence.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "moderate_risk": [
                            {
                                "type": "medication",
                                "name": "Medication optimization",
                                "description": "Ensure optimal antipsychotic treatment with monitoring for efficacy and side effects.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "therapy",
                                "name": "Cognitive Behavioral Therapy for psychosis",
                                "description": "Evidence-based CBT modified for psychotic symptoms to address delusions and hallucinations.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Social skills training",
                                "description": "Structured training to improve interpersonal skills and social functioning.",
                                "evidence_level": "A",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "support",
                                "name": "Supported employment",
                                "description": "Vocational support using Individual Placement and Support (IPS) model when appropriate.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "support",
                                "name": "Family intervention",
                                "description": "Structured family intervention to reduce expressed emotion and improve support.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            }
                        ],
                        "low_risk": [
                            {
                                "type": "medication",
                                "name": "Medication maintenance",
                                "description": "Continue effective antipsychotic regimen with regular monitoring.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "education",
                                "name": "Ongoing psychoeducation",
                                "description": "Continued education about schizophrenia, medications, and relapse prevention.",
                                "evidence_level": "A",
                                "recommendation_strength": "Strong"
                            },
                            {
                                "type": "skills",
                                "name": "Cognitive remediation",
                                "description": "Consider cognitive training to address cognitive deficits if present.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "lifestyle",
                                "name": "Healthy lifestyle promotion",
                                "description": "Encourage physical activity, proper nutrition, and smoking cessation.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            },
                            {
                                "type": "support",
                                "name": "Community integration",
                                "description": "Support meaningful community participation and social connection.",
                                "evidence_level": "B",
                                "recommendation_strength": "Moderate"
                            }
                        ]
                    },
                    "assessment_tools": [
                        {
                            "name": "PANSS",
                            "description": "Positive and Negative Syndrome Scale",
                            "cutoffs": {
                                "absent": [30, 58],
                                "mild": [59, 94],
                                "moderate": [95, 130],
                                "severe": [131, 210]
                            }
                        },
                        {
                            "name": "BPRS",
                            "description": "Brief Psychiatric Rating Scale",
                            "cutoffs": {
                                "not ill": [0, 31],
                                "minimally ill": [32, 41],
                                "moderately ill": [42, 52],
                                "markedly ill": [53, 126]
                            }
                        }
                    ],
                    "comorbidities": [
                        {
                            "condition": "Substance Use",
                            "adjustment": "Integrated dual diagnosis treatment strongly recommended."
                        },
                        {
                            "condition": "Depression",
                            "adjustment": "Differentiate from negative symptoms; may need specific antidepressant treatment."
                        },
                        {
                            "condition": "Metabolic Issues",
                            "adjustment": "Regular monitoring of weight, glucose, and lipids; lifestyle interventions."
                        }
                    ]
                }
            },
            "general_recommendations": [
                {
                    "name": "Regular follow-up",
                    "description": "Maintain regular clinical follow-up appointments to monitor progress and adjust treatment as needed.",
                    "evidence_level": "A",
                    "applies_to": "all"
                },
                {
                    "name": "Treatment adherence",
                    "description": "Emphasize the importance of adhering to recommended treatments, including medication and therapy.",
                    "evidence_level": "A",
                    "applies_to": "all"
                },
                {
                    "name": "Crisis plan",
                    "description": "Develop a crisis plan with steps to take if symptoms worsen significantly.",
                    "evidence_level": "B",
                    "applies_to": "all"
                }
            ],
            "evidence_levels": {
                "A": "Multiple randomized controlled trials with consistent findings",
                "B": "At least one randomized controlled trial or multiple observational studies",
                "C": "Expert consensus or case series"
            },
            "recommendation_strengths": {
                "Strong": "Benefits clearly outweigh risks; recommended for most patients",
                "Moderate": "Benefits likely outweigh risks; consider for many patients",
                "Weak": "Balance of benefits and risks unclear; consider on individual basis"
            }
        }
    
    def get_recommendations(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on assessment data.
        
        Args:
            assessment_data (dict): Assessment data including risk scores
            
        Returns:
            dict: Recommendations
        """
        recommendations = {
            "primary": [],
            "secondary": [],
            "general": [],
            "personalized_message": "",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Get risk scores
        risk_scores = assessment_data.get("risk_scores", {})
        
        if not risk_scores:
            # No risk scores to base recommendations on
            recommendations["personalized_message"] = "Unable to generate specific recommendations without risk scores."
            self._add_general_recommendations(recommendations)
            return recommendations
        
        # Find the primary condition (highest risk score)
        primary_condition = max(risk_scores.items(), key=lambda x: x[1])
        condition_name = primary_condition[0]
        risk_score = primary_condition[1]
        
        # Add personalized message
        recommendations["personalized_message"] = self._generate_personalized_message(condition_name, risk_score)
        
        # Get risk level
        risk_level = self._get_risk_level(risk_score)
        
        # Get recommendations for primary condition
        self._add_condition_recommendations(recommendations, condition_name, risk_level, primary=True)
        
        # Add recommendations for other conditions with high risk scores
        for condition, score in risk_scores.items():
            if condition != condition_name and score > 0.3:
                self._add_condition_recommendations(recommendations, condition, self._get_risk_level(score), primary=False)
        
        # Add general recommendations
        self._add_general_recommendations(recommendations)
        
        return recommendations
    
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level from risk score.
        
        Args:
            risk_score (float): Risk score (0-1)
            
        Returns:
            str: Risk level (high_risk, moderate_risk, low_risk)
        """
        if risk_score >= 0.7:
            return "high_risk"
        elif risk_score >= 0.4:
            return "moderate_risk"
        else:
            return "low_risk"
    
    def _add_condition_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]],
                                     condition: str, risk_level: str, primary: bool = False):
        """
        Add condition-specific recommendations.
        
        Args:
            recommendations (dict): Recommendations object to update
            condition (str): Condition name
            risk_level (str): Risk level
            primary (bool): Whether this is the primary condition
        """
        # Check if condition exists in knowledge base
        if condition not in self.knowledge_base["conditions"]:
            return
        
        # Get interventions for this condition and risk level
        condition_data = self.knowledge_base["conditions"][condition]
        interventions = condition_data.get("interventions", {}).get(risk_level, [])
        
        # Limit to top recommendations
        top_limit = 5 if primary else 3
        if len(interventions) > top_limit:
            # Prioritize by recommendation strength and evidence level
            def get_priority(intervention):
                strength_priority = {"Strong": 3, "Moderate": 2, "Weak": 1}
                evidence_priority = {"A": 3, "B": 2, "C": 1}
                
                return (
                    strength_priority.get(intervention.get("recommendation_strength", "Weak"), 0),
                    evidence_priority.get(intervention.get("evidence_level", "C"), 0)
                )
            
            # Sort by priority and take top recommendations
            interventions = sorted(interventions, key=get_priority, reverse=True)[:top_limit]
        
        # Add to appropriate recommendation list
        target_list = "primary" if primary else "secondary"
        
        for intervention in interventions:
            recommendation = {
                "condition": condition,
                "risk_level": risk_level,
                "intervention_type": intervention.get("type", ""),
                "name": intervention.get("name", ""),
                "description": intervention.get("description", ""),
                "evidence_level": intervention.get("evidence_level", ""),
                "recommendation_strength": intervention.get("recommendation_strength", "")
            }
            
            recommendations[target_list].append(recommendation)
    
    def _add_general_recommendations(self, recommendations: Dict[str, List[Dict[str, Any]]]):
        """
        Add general recommendations.
        
        Args:
            recommendations (dict): Recommendations object to update
        """
        general_recs = self.knowledge_base.get("general_recommendations", [])
        
        for rec in general_recs:
            recommendation = {
                "name": rec.get("name", ""),
                "description": rec.get("description", ""),
                "evidence_level": rec.get("evidence_level", "")
            }
            
            recommendations["general"].append(recommendation)
    
    def _generate_personalized_message(self, condition: str, risk_score: float) -> str:
        """
        Generate a personalized message based on condition and risk score.
        
        Args:
            condition (str): Condition name
            risk_score (float): Risk score
            
        Returns:
            str: Personalized message
        """
        risk_level = self._get_risk_level(risk_score)
        risk_text = "high" if risk_level == "high_risk" else "moderate" if risk_level == "moderate_risk" else "low"
        
        messages = {
            "high_risk": [
                f"Your assessment indicates a high risk for {condition}. It's important to seek professional help promptly.",
                f"We've detected significant indicators of {condition}. The following recommendations should be prioritized.",
                f"Based on your responses, there are concerning signs of {condition} that require prompt attention."
            ],
            "moderate_risk": [
                f"Your assessment shows a moderate risk for {condition}. The following recommendations may help.",
                f"We've identified some indicators of {condition} at a moderate level. Consider these recommendations.",
                f"Based on your responses, there are some signs of {condition} that should be addressed."
            ],
            "low_risk": [
                f"Your assessment indicates a low risk for {condition}, but ongoing monitoring is recommended.",
                f"We've detected minimal indicators of {condition}. Consider these preventive recommendations.",
                f"Based on your responses, there are mild signs of {condition}. The following suggestions may be helpful."
            ]
        }
        
        # Select a random message from the appropriate list
        import random
        return random.choice(messages[risk_level])


class TreatmentPlanner:
    """Class for generating comprehensive treatment plans."""
    
    def __init__(self, recommendation_engine: RecommendationEngine):
        """
        Initialize the treatment planner.
        
        Args:
            recommendation_engine (RecommendationEngine): Recommendation engine
        """
        self.recommendation_engine = recommendation_engine
    
    def create_treatment_plan(self, assessment_data: Dict[str, Any], 
                            patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive treatment plan.
        
        Args:
            assessment_data (dict): Assessment data
            patient_info (dict, optional): Patient information
            
        Returns:
            dict: Treatment plan
        """
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(assessment_data)
        
        # Create treatment plan structure
        treatment_plan = {
            "patient_id": assessment_data.get("patient_id", ""),
            "assessment_id": assessment_data.get("id", ""),
            "plan_date": datetime.datetime.now().isoformat(),
            "summary": self._generate_summary(assessment_data, recommendations),
            "goals": self._generate_goals(assessment_data, recommendations),
            "interventions": self._organize_interventions(recommendations),
            "medications": self._generate_medication_recommendations(assessment_data, recommendations),
            "monitoring": self._generate_monitoring_plan(assessment_data, recommendations),
            "follow_up": self._generate_follow_up_plan(assessment_data, recommendations),
            "references": self._generate_references(recommendations)
        }
        
        return treatment_plan
    
    def _generate_summary(self, assessment_data: Dict[str, Any], 
                         recommendations: Dict[str, Any]) -> str:
        """
        Generate a summary of the treatment plan.
        
        Args:
            assessment_data (dict): Assessment data
            recommendations (dict): Recommendations
            
        Returns:
            str: Summary
        """
        risk_scores = assessment_data.get("risk_scores", {})
        
        if not risk_scores:
            return "Treatment plan based on clinical assessment. No specific risk scores available."
        
        # Find primary condition
        primary_condition = max(risk_scores.items(), key=lambda x: x[1])
        condition = primary_condition[0]
        score = primary_condition[1]
        
        # Create summary
        risk_level = "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"
        
        summary = f"Treatment plan targeting {condition} (risk level: {risk_level})."
        
        # Add secondary conditions if applicable
        secondary_conditions = [c for c, s in risk_scores.items() if c != condition and s > 0.3]
        if secondary_conditions:
            summary += f" Also addressing: {', '.join(secondary_conditions)}."
        
        # Add recommended approaches
        primary_interventions = [r["name"] for r in recommendations["primary"][:3]]
        if primary_interventions:
            summary += f" Key interventions include: {', '.join(primary_interventions)}."
        
        return summary
    
    def _generate_goals(self, assessment_data: Dict[str, Any], 
                       recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate treatment goals.
        
        Args:
            assessment_data (dict): Assessment data
            recommendations (dict): Recommendations
            
        Returns:
            list: Treatment goals
        """
        goals = []
        
        # Get risk scores
        risk_scores = assessment_data.get("risk_scores", {})
        
        # Create condition-specific goals
        for condition, score in risk_scores.items():
            if score > 0.3:
                risk_level = "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"
                
                if condition == "Depression":
                    goals.append({
                        "domain": "Symptoms",
                        "target": f"Reduce {condition.lower()} symptoms",
                        "measure": "PHQ-9 score",
                        "timeframe": "4-6 weeks",
                        "objective": "50% reduction in symptoms"
                    })
                    
                    goals.append({
                        "domain": "Functioning",
                        "target": "Improve daily functioning",
                        "measure": "Self-reported activity level",
                        "timeframe": "8-12 weeks",
                        "objective": "Return to baseline functioning in work/social domains"
                    })
                    
                elif condition == "Anxiety":
                    goals.append({
                        "domain": "Symptoms",
                        "target": f"Reduce {condition.lower()} symptoms",
                        "measure": "GAD-7 score",
                        "timeframe": "4-6 weeks",
                        "objective": "50% reduction in symptoms"
                    })
                    
                    goals.append({
                        "domain": "Coping",
                        "target": "Improve anxiety management skills",
                        "measure": "Self-reported skill use",
                        "timeframe": "4-8 weeks",
                        "objective": "Daily use of at least 2 anxiety management strategies"
                    })
                    
                elif condition == "PTSD":
                    goals.append({
                        "domain": "Symptoms",
                        "target": f"Reduce {condition.lower()} symptoms",
                        "measure": "PCL-5 score",
                        "timeframe": "8-12 weeks",
                        "objective": "Clinically significant reduction in symptoms"
                    })
                    
                    goals.append({
                        "domain": "Safety",
                        "target": "Improve emotional regulation and distress tolerance",
                        "measure": "Self-reported distress levels",
                        "timeframe": "4-8 weeks",
                        "objective": "Ability to manage trauma-related distress without unsafe behaviors"
                    })
                    
                elif condition == "Bipolar":
                    goals.append({
                        "domain": "Mood Stability",
                        "target": "Achieve mood stabilization",
                        "measure": "Mood chart ratings",
                        "timeframe": "8-12 weeks",
                        "objective": "Absence of manic/hypomanic episodes and reduction in mood fluctuations"
                    })
                    
                    goals.append({
                        "domain": "Routine",
                        "target": "Establish regular sleep and activity patterns",
                        "measure": "Sleep log",
                        "timeframe": "4-6 weeks",
                        "objective": "Consistent sleep-wake times with 7-9 hours of sleep nightly"
                    })
                    
                elif condition == "Schizophrenia":
                    goals.append({
                        "domain": "Positive Symptoms",
                        "target": "Reduce positive symptoms (hallucinations, delusions)",
                        "measure": "PANSS positive subscale",
                        "timeframe": "8-12 weeks",
                        "objective": "Clinically significant reduction in positive symptoms"
                    })
                    
                    goals.append({
                        "domain": "Functioning",
                        "target": "Improve community functioning",
                        "measure": "Independent living skills",
                        "timeframe": "12-16 weeks",
                        "objective": "Improvement in self-care and community navigation skills"
                    })
        
        # Add general goals if specific ones couldn't be generated
        if not goals:
            goals = [
                {
                    "domain": "Symptoms",
                    "target": "Reduce mental health symptoms",
                    "measure": "Standardized assessments",
                    "timeframe": "8-12 weeks",
                    "objective": "Clinically significant improvement in symptoms"
                },
                {
                    "domain": "Functioning",
                    "target": "Improve overall functioning",
                    "measure": "Self-reported functioning",
                    "timeframe": "12-16 weeks",
                    "objective": "Return to baseline functioning in major life domains"
                }
            ]
        
        return goals
    
    def _organize_interventions(self, recommendations: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize interventions by type.
        
        Args:
            recommendations (dict): Recommendations
            
        Returns:
            dict: Organized interventions
        """
        intervention_types = {
            "therapy": [],
            "medication": [],
            "skills": [],
            "lifestyle": [],
            "support": [],
            "monitoring": []
        }
        
        # Process primary recommendations
        for rec in recommendations.get("primary", []):
            intervention_type = rec.get("intervention_type", "")
            
            if intervention_type in intervention_types:
                intervention_types[intervention_type].append({
                    "name": rec.get("name", ""),
                    "description": rec.get("description", ""),
                    "evidence_level": rec.get("evidence_level", ""),
                    "priority": "High",
                    "for_condition": rec.get("condition", "")
                })
        
        # Process secondary recommendations
        for rec in recommendations.get("secondary", []):
            intervention_type = rec.get("intervention_type", "")
            
            if intervention_type in intervention_types:
                intervention_types[intervention_type].append({
                    "name": rec.get("name", ""),
                    "description": rec.get("description", ""),
                    "evidence_level": rec.get("evidence_level", ""),
                    "priority": "Medium",
                    "for_condition": rec.get("condition", "")
                })
        
        # Add general recommendations
        for rec in recommendations.get("general", []):
            # Categorize general recommendations
            name = rec.get("name", "").lower()
            
            if "medication" in name or "treatment" in name:
                category = "medication"
            elif "follow-up" in name or "monitor" in name:
                category = "monitoring"
            elif "support" in name or "crisis" in name:
                category = "support"
            else:
                category = "lifestyle"
            
            intervention_types[category].append({
                "name": rec.get("name", ""),
                "description": rec.get("description", ""),
                "evidence_level": rec.get("evidence_level", ""),
                "priority": "Medium",
                "for_condition": "General"
            })
        
        # Remove empty categories
        return {k: v for k, v in intervention_types.items() if v}
    
    def _generate_medication_recommendations(self, assessment_data: Dict[str, Any], 
                                           recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate medication recommendations.
        
        Args:
            assessment_data (dict): Assessment data
            recommendations (dict): Recommendations
            
        Returns:
            list: Medication recommendations
        """
        medications = []
        
        # Extract medication recommendations from interventions
        for rec in recommendations.get("primary", []) + recommendations.get("secondary", []):
            if rec.get("intervention_type") == "medication":
                medications.append({
                    "recommendation": rec.get("name", ""),
                    "details": rec.get("description", ""),
                    "evidence_level": rec.get("evidence_level", ""),
                    "for_condition": rec.get("condition", ""),
                    "priority": "High" if rec in recommendations.get("primary", []) else "Medium"
                })
        
        # If no specific medications are recommended, add a general note
        if not medications:
            medications.append({
                "recommendation": "Medication evaluation",
                "details": "Consider medication evaluation by a psychiatrist based on symptom profile and severity.",
                "evidence_level": "B",
                "for_condition": "General",
                "priority": "Medium"
            })
        
        return medications
    
    def _generate_monitoring_plan(self, assessment_data: Dict[str, Any], 
                                recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a monitoring plan.
        
        Args:
            assessment_data (dict): Assessment data
            recommendations (dict): Recommendations
            
        Returns:
            list: Monitoring plan
        """
        monitoring = []
        risk_scores = assessment_data.get("risk_scores", {})
        
        # Add condition-specific monitoring
        for condition, score in risk_scores.items():
            if score > 0.3:
                if condition == "Depression":
                    monitoring.append({
                        "target": "Depression symptoms",
                        "measure": "PHQ-9",
                        "frequency": "Every 2-4 weeks",
                        "threshold": "PHQ-9 score increase of ≥5 points or total score ≥15"
                    })
                    
                    monitoring.append({
                        "target": "Suicidal ideation",
                        "measure": "Columbia-Suicide Severity Rating Scale (C-SSRS)",
                        "frequency": "Every visit",
                        "threshold": "Any active suicidal ideation with intent or plan"
                    })
                    
                elif condition == "Anxiety":
                    monitoring.append({
                        "target": "Anxiety symptoms",
                        "measure": "GAD-7",
                        "frequency": "Every 2-4 weeks",
                        "threshold": "GAD-7 score increase of ≥5 points or total score ≥15"
                    })
                    
                elif condition == "PTSD":
                    monitoring.append({
                        "target": "PTSD symptoms",
                        "measure": "PCL-5",
                        "frequency": "Every 4 weeks",
                        "threshold": "PCL-5 score increase of ≥10 points"
                    })
                    
                elif condition == "Bipolar":
                    monitoring.append({
                        "target": "Mood episodes",
                        "measure": "Daily mood chart",
                        "frequency": "Daily",
                        "threshold": "Mood elevation ≥7 or depression ≤3 for ≥3 consecutive days"
                    })
                    
                    monitoring.append({
                        "target": "Sleep patterns",
                        "measure": "Sleep log",
                        "frequency": "Daily",
                        "threshold": "Sleep <5 hours for 2 consecutive nights"
                    })
                    
                elif condition == "Schizophrenia":
                    monitoring.append({
                        "target": "Psychotic symptoms",
                        "measure": "Brief Psychiatric Rating Scale (BPRS)",
                        "frequency": "Every 4 weeks",
                        "threshold": "BPRS score increase of ≥10 points"
                    })
                    
                    monitoring.append({
                        "target": "Medication side effects",
                        "measure": "AIMS/Barnes scales",
                        "frequency": "Every 3 months",
                        "threshold": "Any abnormal involuntary movements"
                    })
        
        # Add general monitoring
        monitoring.append({
            "target": "Treatment adherence",
            "measure": "Self-report and pill count",
            "frequency": "Every visit",
            "threshold": "Missing >20% of medication doses or therapy appointments"
        })
        
        monitoring.append({
            "target": "Overall functioning",
            "measure": "WHO Disability Assessment Schedule (WHODAS 2.0)",
            "frequency": "Every 3 months",
            "threshold": "Significant deterioration in functioning"
        })
        
        return monitoring
    
    def _generate_follow_up_plan(self, assessment_data: Dict[str, Any], 
                               recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a follow-up plan.
        
        Args:
            assessment_data (dict): Assessment data
            recommendations (dict): Recommendations
            
        Returns:
            dict: Follow-up plan
        """
        # Determine risk level from highest risk score
        risk_scores = assessment_data.get("risk_scores", {})
        max_risk = max(risk_scores.values()) if risk_scores else 0
        
        if max_risk >= 0.7:
            initial_follow_up = "Within 1 week"
            subsequent_frequency = "Weekly for first month, then biweekly"
            provider = "Psychiatrist and therapist"
        elif max_risk >= 0.4:
            initial_follow_up = "Within 2 weeks"
            subsequent_frequency = "Biweekly for first month, then monthly"
            provider = "Mental health provider (psychiatrist or therapist)"
        else:
            initial_follow_up = "Within 3-4 weeks"
            subsequent_frequency = "Monthly"
            provider = "Mental health provider or primary care with mental health focus"
        
        return {
            "initial_follow_up": initial_follow_up,
            "subsequent_frequency": subsequent_frequency,
            "recommended_provider": provider,
            "monitoring_expectations": "Review symptoms, medication effects/side effects, and treatment adherence",
            "escalation_criteria": "Significant symptom worsening, emergence of safety concerns, or lack of expected improvement"
        }
    
    def _generate_references(self, recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate references.
        
        Args:
            recommendations (dict): Recommendations
            
        Returns:
            list: References
        """
        # Create generic references based on evidence levels
        references = []
        unique_refs = set()
        
        # Add references from primary recommendations with high evidence
        for rec in recommendations.get("primary", []):
            evidence = rec.get("evidence_level", "")
            condition = rec.get("condition", "")
            name = rec.get("name", "")
            
            if evidence == "A" and (condition, name) not in unique_refs:
                references.append({
                    "title": f"Clinical Guidelines for {condition}",
                    "source": "Medical Association Guidelines",
                    "year": "2023",
                    "evidence_level": "A",
                    "relevant_for": f"{name} for {condition}"
                })
                unique_refs.add((condition, name))
        
        # Add generic evidence-based references
        generic_references = [
            {
                "title": "American Psychiatric Association Practice Guidelines",
                "source": "American Psychiatric Association",
                "year": "2022",
                "evidence_level": "A",
                "relevant_for": "General treatment recommendations"
            },
            {
                "title": "National Institute for Health and Care Excellence (NICE) Guidelines",
                "source": "NICE",
                "year": "2023",
                "evidence_level": "A",
                "relevant_for": "Evidence-based mental health interventions"
            },
            {
                "title": "World Health Organization Mental Health Gap Action Programme (mhGAP)",
                "source": "WHO",
                "year": "2021",
                "evidence_level": "A",
                "relevant_for": "Global standards for mental health care"
            }
        ]
        
        # Add generic references if needed
        for ref in generic_references:
            if len(references) < 5:  # Limit to 5 total references
                references.append(ref)
        
        return references
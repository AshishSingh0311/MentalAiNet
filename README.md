# MentalAiNet: Multimodal Mental Health Diagnostics Framework

A comprehensive framework for multimodal deep learning in mental health diagnostics with explainable AI capabilities.

## Features

- **Multimodal Data Processing**
  - Text Analysis (clinical notes, social media content)
  - Audio Processing (speech patterns, tone analysis)
  - Physiological Signals (ECG, EEG)
  - Medical Imaging (fMRI, MRI)

- **AI/ML Capabilities**
  - Deep Learning Model Training
  - Real-time Assessment
  - Risk Score Calculation
  - Explainable AI Techniques

- **Clinical Tools**
  - Patient Assessment Dashboard
  - Risk Score Visualization
  - Treatment Planning
  - Clinical Recommendations

- **Data Management**
  - Secure Patient Records
  - Assessment History
  - Data Export (PDF, HTML, CSV)
  - Batch Processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AshishSingh0311/MentalAiNet.git
cd MentalAiNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
python setup_database.py
```

## Usage

Run the Streamlit application:
```bash
streamlit run simplified_app.py
```

The application will be available at http://localhost:8501

## Project Structure

```
MentalAiNet/
├── modules/
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessor.py     # Data preprocessing
│   ├── models.py          # Model definitions
│   ├── training.py        # Training utilities
│   ├── evaluation.py      # Model evaluation
│   ├── explainability.py  # AI explainability
│   ├── visualization.py   # Visualization tools
│   └── database.py        # Database management
├── utils/
│   ├── export.py          # Export utilities
│   └── helpers.py         # Helper functions
├── simplified_app.py      # Main Streamlit application
└── setup_database.py      # Database setup script
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
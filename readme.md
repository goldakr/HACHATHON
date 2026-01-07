# Postpartum Depression (PPD) Prediction System
## AI-Powered Risk Assessment & Conversational Screening Tool

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Dataset](#dataset)
6. [Machine Learning Models](#machine-learning-models)
7. [User Interface](#user-interface)
8. [Installation & Setup](#installation--setup)
9. [Usage](#usage)
10. [Project Structure](#project-structure)
11. [Key Components](#key-components)
12. [Future Enhancements](#future-enhancements)

---

## 🎯 Project Overview

This project is an **AI-powered Postpartum Depression (PPD) Risk Prediction System** that combines:

- **Machine Learning Models** (XGBoost & Random Forest) for accurate risk assessment
- **SHAP Explainable AI** for transparent, interpretable predictions
- **EPDS Conversational Agent** using LangChain for natural language screening
- **Interactive Gradio Interface** for easy-to-use clinical assessment tools
- **Domain Knowledge Integration** for medically-informed risk adjustments

The system helps healthcare professionals and researchers identify women at risk for postpartum depression by analyzing demographic, clinical, psychiatric, and psychosocial factors.

---

## ✨ Key Features

### 1. **Dual ML Algorithm Support**
   - **XGBoost Classifier** with hyperparameter optimization
   - **Random Forest Classifier** with default parameters
   - Automatic algorithm selection and comparison
   - Model persistence and retraining capabilities

### 2. **Explainable AI (XAI)**
   - **SHAP (SHapley Additive exPlanations)** integration
   - Feature importance visualization
   - Personalized risk explanations
   - Waterfall-style SHAP plots for individual predictions

### 3. **EPDS Conversational Agent**
   - **LangChain-powered** natural language interaction
   - **Hebrew language support** for patient-friendly conversations
   - **Sentiment analysis** using TextBlob and LLM-based distress detection
   - **Natural language understanding** for patient responses
   - Automatic EPDS score calculation
   - Crisis resource provision for high-risk patients

### 4. **Comprehensive Visualizations**
   - Target distribution plots
   - Feature distributions by target class
   - Confusion matrix
   - ROC curve with AUC score
   - Prediction probability distributions
   - Correlation heatmaps
   - SHAP summary plots

### 5. **Domain Knowledge Integration**
   - Medical rule-based adjustments
   - EPDS score integration
   - Risk stratification based on clinical factors
   - Context-aware probability corrections

### 6. **User-Friendly Interface**
   - **Gradio web interface** with tabbed navigation
   - Real-time predictions
   - Interactive visualizations
   - Model training controls
   - Agent chat interfaces (General AI & EPDS)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Training   │  │  Prediction  │  │Visualizations│     │
│  │    Tab       │  │     Tab      │  │     Tab      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Medical Staff Chatbot  │  │ EPDS Chatbot │                        │
│  │     Tab      │  │     Tab      │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      PPD Agent Layer                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  • Model Prediction                                 │    │
│  │  • SHAP Explanations                                │    │
│  │  • Domain Knowledge Rules                           │    │
│  │  • Feature Engineering                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Machine Learning Models                    │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   XGBoost    │              │   Random     │            │
│  │  Classifier  │              │   Forest     │            │
│  │              │              │  Classifier  │            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  • Demographics                                             │
│  • Clinical Data                                            │
│  • Psychiatric History                                      │
│  • Functional & Psychosocial Data                           │
│  • EPDS Assessment Data                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning pipeline and preprocessing

### Machine Learning
- **XGBoost** - Gradient boosting classifier
- **Random Forest** - Ensemble learning classifier
- **SHAP** - Explainable AI library

### AI & NLP
- **LangChain** - Agent framework for conversational AI
- **OpenAI API** - Large Language Model integration
- **TextBlob** - Sentiment analysis

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualization
- **Gradio** - Web interface framework

### Additional Libraries
- **Python-dotenv** - Environment variable management
- **Pathlib** - Path handling

---

## 📊 Dataset

The system uses a comprehensive dataset combining multiple sources:

### Data Sources
1. **Demographics.csv** - Age, marital status, SES, population, employment
2. **Clinical_data.csv** - First birth, GDM, TSH, NVP, GH, mode of birth
3. **Psychiatric_data.csv** - Depression/anxiety history, medication use, PPD target
4. **Functional_Psychosocial_data.csv** - Sleep quality, fatigue, partner support, family support, domestic violence
5. **EPDS_answers.csv** - EPDS total scores and self-harm thoughts

### Data Processing
- **Inner joins** on patient ID across all tables
- **Missing value handling** - rows with missing values are dropped
- **Target encoding** - PPD column converted from "Yes"/"No" to binary (1/0)
- **Feature engineering** - Categorical encoding via OneHotEncoder
- **Train/test split** - 80/20 split with stratification

---

## 🤖 Machine Learning Models

### Model Options

#### 1. XGBoost Classifier
- **Hyperparameter optimization** via RandomizedSearchCV
- **Parameters optimized:**
  - `n_estimators`: 100-300
  - `max_depth`: 3-10
  - `learning_rate`: 0.01-0.3
  - `subsample`: 0.8-1.0
  - `colsample_bytree`: 0.8-1.0
- **Evaluation metric:** ROC AUC score

#### 2. Random Forest Classifier
- **Default parameters** (optimization available but not implemented in UI)
- **n_estimators:** 100
- **max_depth:** None (unlimited)
- **Random state:** 42 for reproducibility

### Model Performance
- Models are evaluated using:
  - **ROC AUC Score**
  - **Confusion Matrix**
  - **Classification Report**
- Performance metrics are displayed in the interface

---

## 💻 User Interface

### Gradio Interface Tabs

#### 1. **Training Tab**
   - Algorithm selection (XGBoost/Random Forest)
   - Hyperparameter optimization toggle
   - Training status and performance metrics
   - Model save/load functionality

#### 2. **Prediction Tab**
   - Patient data input form
   - Real-time risk prediction
   - Risk percentage and level
   - Top 5 feature importance
   - Personalized explanation
   - SHAP visualization

#### 3. **Visualizations Tab**
   - Target distribution
   - Feature distributions
   - Confusion matrix
   - ROC curve
   - Prediction distribution
   - Correlation heatmap
   - SHAP summary plots

#### 4. **Medical Staff Chatbot Tab**
   - LangChain-powered conversational agent
   - Natural language queries about PPD risk
   - Example questions provided
   - Integration with PPD prediction model

#### 5. **EPDS Chatbot Tab**
   - Edinburgh Postnatal Depression Scale conversational assessment
   - Natural language patient interaction
   - Sentiment analysis and distress detection
   - Automatic scoring and risk assessment
   - Crisis resource provision

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
cd Hackathon
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables (Optional - for LangChain features)
Create a `.env` file in the project root (one level up from Hackathon):
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5: Verify Data Files
Ensure the following CSV files are in the `data/` directory:
- `Demographics.csv`
- `Clinical_data.csv`
- `Psychiatric_data.csv`
- `Functional_Psychosocial_data.csv`
- `EPDS_answers.csv`

---

## 📖 Usage

### Running the Application

1. **Start the Gradio Interface:**
   ```bash
   python main.py
   ```

2. **Access the Interface:**
   - The application will start a local web server
   - Open your browser to the URL shown (typically `http://127.0.0.1:7860`)

3. **Train a Model:**
   - Navigate to the "Training" tab
   - Select your preferred algorithm (XGBoost or Random Forest)
   - Optionally enable hyperparameter optimization
   - Click "Start Train Model"
   - Wait for training to complete

4. **Make Predictions:**
   - Navigate to the "Prediction" tab
   - Fill in patient information
   - Click "Assess Risk"
   - View predictions, explanations, and visualizations

5. **Use Chatbots:**
   - **Medical Staff Chatbot:** Ask questions about PPD risk assessment
   - **EPDS Chatbot:** Conduct conversational EPDS screening with patients

### Loading Pre-trained Models

The system automatically loads pre-trained models from:
- `output/agents/ppd_agent_xgboost.pkl` (XGBoost model)
- `output/agents/ppd_agent_rf.pkl` (Random Forest model)

If these files exist, the interface will load them on startup.

---

## 📁 Project Structure

```
Hackathon/
│
├── data/                          # Dataset files
│   ├── Demographics.csv
│   ├── Clinical_data.csv
│   ├── Psychiatric_data.csv
│   ├── Functional_Psychosocial_data.csv
│   └── EPDS_answers.csv
│
├── output/                        # Generated outputs
│   ├── agents/                    # Saved model agents
│   │   ├── ppd_agent_xgboost.pkl
│   │   └── ppd_agent_rf.pkl
│   └── plots/                     # Generated visualizations
│       ├── XGBoost/
│       └── RandomForest/
│
├── main.py                        # Main entry point
├── gradio_app.py                  # Gradio interface definition
├── ppd_agent.py                   # PPD Agent class
├── MLmodel.py                     # ML model creation and training
├── visualization.py               # Plot generation functions
├── gradio_visualizations.py       # Gradio-specific visualizations
├── gradio_predictions.py          # Prediction functions
├── gradio_helpers.py              # Helper functions
├── epds_agent.py                  # EPDS conversational agent
├── EPDS_questions.py              # EPDS question definitions
├── langchain_tool.py              # LangChain tool integration
├── exceptions.py                  # Custom exception classes
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔧 Key Components

### 1. PPDAgent Class (`ppd_agent.py`)
   - Core agent class for PPD predictions
   - SHAP explainer integration
   - Domain knowledge rule application
   - Model training methods (XGBoost & Random Forest)
   - Model persistence (save/load)

### 2. Gradio Interface (`gradio_app.py`)
   - Complete web interface definition
   - Training wrapper functions
   - Prediction handlers
   - Visualization loaders
   - Chatbot integrations

### 3. EPDS Agent (`epds_agent.py`)
   - Conversational agent for EPDS assessment
   - LangChain integration
   - Sentiment analysis
   - Natural language understanding
   - Hebrew language support

### 4. Visualization Module (`visualization.py`)
   - Plot generation functions
   - Base64 encoding for web display
   - Plot saving functionality

### 5. ML Model Module (`MLmodel.py`)
   - Pipeline creation (XGBoost & Random Forest)
   - Hyperparameter optimization
   - Model training and evaluation

---

## 📈 Future Enhancements

### Planned Improvements
- [ ] Additional ML algorithms (SVM, Neural Networks)
- [ ] Real-time data streaming integration
- [ ] Multi-language support expansion
- [ ] Mobile app interface
- [ ] Electronic Health Record (EHR) integration
- [ ] Longitudinal risk tracking
- [ ] Clinician dashboard
- [ ] Patient portal
- [ ] Automated report generation
- [ ] API deployment (FastAPI server available in `api_server.py`)

### Research Opportunities
- Feature importance analysis across different populations
- Model interpretability studies
- Clinical validation studies
- Integration with wearable device data
- Multi-modal data fusion (text, clinical, genetic)

---

## 📝 Notes

### Model Persistence
- Models are saved automatically after training
- Saved models can be loaded on application startup
- Models are algorithm-specific (separate files for XGBoost and Random Forest)

### Path Handling
- All paths are resolved relative to the Hackathon directory
- Uses `pathlib.Path` for cross-platform compatibility
- Outputs are saved to `output/` directory structure

### Environment Variables
- LangChain features require `OPENAI_API_KEY` in `.env` file
- System works in "basic mode" without API key (no chatbot features)

---

## 👥 Acknowledgments

This project was developed as part of PPD Prediction Prototype, focusing on:
- Machine learning model development
- Explainable AI (XAI) implementation
- Natural language processing
- Clinical decision support systems
- User interface design

---

## 📄 License

This project is for educational and research purposes.

---

## 📧 Contact

For questions or contributions, please refer to the project repository.

---

**Last Updated:** 2024

**Version:** 1.0

---

## 🎓 PowerPoint Presentation Highlights

### Suggested Slides:

1. **Title Slide:** Postpartum Depression Prediction System
2. **Problem Statement:** Need for early PPD detection
3. **Solution Overview:** AI-powered risk assessment tool
4. **System Architecture:** Diagram showing components
5. **Data Sources:** Multi-source dataset integration
6. **ML Models:** XGBoost vs Random Forest comparison
7. **Explainable AI:** SHAP visualizations and interpretations
8. **EPDS Conversational Agent:** Natural language screening
9. **User Interface:** Gradio interface screenshots
10. **Results & Performance:** Model metrics and evaluation
11. **Key Features:** Feature highlights
12. **Future Work:** Planned enhancements
13. **Conclusion:** Summary and impact

---

**Ready for Presentation! 🎤**

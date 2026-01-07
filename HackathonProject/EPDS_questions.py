# EPDS questions
# https://www.healthline.com/health/depression/epds-questions
import os
from textblob import TextBlob
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# Set your OpenAI key in the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Create an OpenAI chat model instance
chat_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    )

agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """את סוכנת רגשית תומכת ליולדות לאחר לידה.
     את מנהלת שיחה קצרה, רגישה ולא פולשנית.
     שאלי שאלות EPDS אחת-אחת.
     לאחר סיום השאלון, בקשי משפט חופשי.
     אל תיתני אבחנה רפואית.
     השתמשי בכלים כאשר מתאים.
     """),
    ("human", "{input}")
])

# Define the EPDS questions
EPDS_QUESTIONS = [
    "בשבוע האחרון, הצלחתי לצחוק ולראות את הצד המצחיק של דברים (0–3)",
    "ציפיתי בהנאה לדברים (0–3)",
    "האשמתי את עצמי ללא סיבה (0–3)",
    "הרגשתי חרדה או דאגה ללא סיבה (0–3)",
    "הרגשתי מפוחדת או מבוהלת (0–3)",
    "הרגשתי שהכול קשה לי מדי (0–3)",
    "היה לי קשה לישון בגלל דאגות (0–3)",
    "הרגשתי עצובה או אומללה (0–3)",
    "הייתי כל כך אומללה שבכיתי (0–3)",
    "עברו בי מחשבות לפגוע בעצמי (0–3)"
]

DISTRESS_KEYWORDS = ["קשה", "עייפה", "בודדה", "לחוצה", "לא מצליחה", "עצובה"]

def analyze_text(text):
    sentiment = TextBlob(text).sentiment.polarity
    keywords_found = [k for k in DISTRESS_KEYWORDS if k in text]
    return sentiment, keywords_found

import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime

# EPDS question column names (matching CSV format)
EPDS_COLUMN_NAMES = [
    "צחוק והצד המצחיק",
    "ציפייה בהנאה",
    "האשמה עצמית",
    "דאגה וחרדה",
    "פחד ובהלה",
    "דברים קשים מדי",
    "קושי לישון",
    "עצב ואומללות",
    "בכי",
    "מחשבות פגיעה עצמית"
]

def save_conversation(name, epds_answers, free_text_answer, sentiment, keywords):
    """Save conversation data to CSV file in the correct format."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    csv_path = data_dir / "EPDS_answers.csv"
    
    # Get next ID
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
            # Get the max ID from existing data
            if 'ID' in existing_df.columns:
                next_id = int(existing_df['ID'].max()) + 1
            else:
                next_id = len(existing_df) + 1
        except Exception as e:
            print(f"⚠️ Error reading existing CSV: {e}")
            next_id = 1
    else:
        next_id = 1
    
    # Create timestamp
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
    
    # Calculate total score
    total_score = sum(epds_answers)
    
    # Ensure we have exactly 10 answers (pad with 0 if needed)
    while len(epds_answers) < 10:
        epds_answers.append(0)
    
    # Create row data matching CSV format
    row_data = {
        "ID": [next_id],
        "Timestamp": [timestamp],
        "Name": [name.strip() if name and name.strip() else f"Patient_{next_id}"],
        "Total Scores": [total_score]
    }
    
    # Add individual question scores
    for i, col_name in enumerate(EPDS_COLUMN_NAMES):
        if i < len(epds_answers):
            row_data[col_name] = [epds_answers[i]]
        else:
            row_data[col_name] = [0]
    
    df = pd.DataFrame(row_data)
    
    # Append to existing file or create new one
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    print(f"✅ Saved EPDS assessment: ID={next_id}, Score={total_score}, Name={name}")
    return next_id, total_score

class EPDSConversation:
    """Manages EPDS conversation state and flow."""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.epds_answers = []
        self.current_q = 0
        self.free_text_answer = ""
        self.patient_name = ""
        self.started = False
        
    def reset(self):
        """Reset conversation state."""
        self.session_id = str(uuid.uuid4())
        self.epds_answers = []
        self.current_q = 0
        self.free_text_answer = ""
        self.patient_name = ""
        self.started = False
    
    def start_conversation(self, name=""):
        """Start a new EPDS conversation."""
        self.reset()
        # Only use UUID if name is truly empty (after stripping whitespace)
        name_clean = name.strip() if name else ""
        self.patient_name = name_clean if name_clean else f"Patient_{self.session_id[:8]}"
        self.started = True
        return f"שלום! אני כאן כדי לעזור לך להעריך את המצב הרגשי שלך לאחר הלידה.\n\n" + \
               f"אנא עני על כל שאלה בסולם של 0-3:\n" + \
               f"0 = בכלל לא\n1 = לא לעתים קרובות\n2 = לפעמים\n3 = לעתים קרובות מאוד\n\n" + \
               f"שאלה 1: {EPDS_QUESTIONS[0]}"
    
    def process_answer(self, user_input):
        """Process user answer and return next question or completion message."""
        try:
            # Try to extract numeric answer (0-3)
            answer = None
            if isinstance(user_input, (int, float)):
                answer = int(user_input)
            elif isinstance(user_input, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    answer = int(numbers[0])
                # Check for Hebrew responses
                elif any(word in user_input.lower() for word in ['לא', 'בכלל לא', '0']):
                    answer = 0
                elif any(word in user_input.lower() for word in ['לעתים', '1']):
                    answer = 1
                elif any(word in user_input.lower() for word in ['לפעמים', '2']):
                    answer = 2
                elif any(word in user_input.lower() for word in ['קרובות', '3']):
                    answer = 3
            
            if answer is None or answer < 0 or answer > 3:
                return "אנא עני עם מספר בין 0-3:\n0 = בכלל לא\n1 = לא לעתים קרובות\n2 = לפעמים\n3 = לעתים קרובות מאוד"
            
            self.epds_answers.append(answer)
            self.current_q += 1
            
            if self.current_q < len(EPDS_QUESTIONS):
                return f"שאלה {self.current_q + 1}: {EPDS_QUESTIONS[self.current_q]}"
            else:
                # All questions answered, ask for free text
                return "רוצה לשתף במשפט אחד איך את מרגישה רגשית לאחר הלידה?"
        
        except Exception as e:
            return f"שגיאה בעיבוד התשובה. אנא נסה שוב: {str(e)}"
    
    def process_free_text(self, user_input):
        """Process free text answer and save conversation."""
        self.free_text_answer = user_input
        sentiment, keywords = analyze_text(user_input)
        record_id, total_score = save_conversation(
            self.patient_name, 
            self.epds_answers, 
            self.free_text_answer, 
            sentiment, 
            keywords
        )
        
        # Determine risk level
        if total_score >= 13:
            risk_level = "גבוה"
            recommendation = "מומלץ לפנות לייעוץ מקצועי"
        elif total_score >= 10:
            risk_level = "בינוני-גבוה"
            recommendation = "מומלץ לעקוב אחרי המצב ולשקול ייעוץ"
        else:
            risk_level = "נמוך-בינוני"
            recommendation = "מומלץ להמשיך לעקוב אחרי המצב"
        
        return f"תודה רבה על השיתוף 💙\n\n" + \
               f"ציון EPDS שלך: {total_score}/30\n" + \
               f"רמת סיכון: {risk_level}\n" + \
               f"{recommendation}\n\n" + \
               f"התשובות נשמרו בהצלחה (רשומה #{record_id})"

# Global conversation instance
epds_conversation = EPDSConversation()
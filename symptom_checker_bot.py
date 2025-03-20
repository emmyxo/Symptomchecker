import re
import logging
import pandas as pd
import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configure paths
BASE_DIR = "C:/Users/USER/Downloads/Spark_Abuja/"
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
SYMPTOM_DESC_PATH = os.path.join(BASE_DIR, "symptom_Description.csv")
SYMPTOM_PRECAUTION_PATH = os.path.join(BASE_DIR, "symptom_precaution.csv")

# Validate required files
required_files = [DATASET_PATH, SYMPTOM_DESC_PATH, SYMPTOM_PRECAUTION_PATH]
for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

class SymptomAnalyzer:
    def __init__(self):
        # Load and process main dataset
        self.diseases = pd.read_csv(DATASET_PATH)
        self.disease_map = self._process_diseases()
        
        # Load supplementary data
        self.descriptions = self._load_supplementary(SYMPTOM_DESC_PATH, 'Description')
        self.precautions = self._load_supplementary(SYMPTOM_PRECAUTION_PATH, 'Precautions')

    def _normalize_name(self, name):
        """Standardize disease names for consistent matching"""
        name = str(name).strip().lower()
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
        name = re.sub(r'\s+', ' ', name)      # Collapse whitespace
        return name

    def _process_diseases(self):
        """Aggregate all symptom variations for each disease"""
        disease_map = {}
        symptom_columns = [col for col in self.diseases.columns if col.startswith('Symptom')]
        
        for _, row in self.diseases.iterrows():
            disease = self._normalize_name(row['Disease'])
            symptoms = {str(row[col]).strip().lower() 
                       for col in symptom_columns 
                       if pd.notna(row[col]) and str(row[col]).strip() != ''}
            
            if disease in disease_map:
                disease_map[disease]['symptoms'].update(symptoms)
                disease_map[disease]['count'] += 1
            else:
                disease_map[disease] = {
                    'symptoms': symptoms,
                    'count': 1,
                    'original_names': set([row['Disease'].strip()])
                }
        
        return disease_map

    def _load_supplementary(self, path, column_name):
        """Load and normalize description/precaution data"""
        df = pd.read_csv(path)
        data_map = {}
        
        for _, row in df.iterrows():
            normalized = self._normalize_name(row['Disease'])
            if normalized not in data_map:
                data_map[normalized] = {
                    'original': row['Disease'].strip(),
                    'data': row[column_name] if column_name == 'Description' else row[1:].tolist()
                }
        return data_map

    def find_matches(self, user_symptoms):
        """Find top 5 matches with match statistics"""
        clean_input = {s.strip().lower() for s in user_symptoms}
        matches = []
        
        for disease, data in self.disease_map.items():
            match_count = len(clean_input & data['symptoms'])
            if match_count > 0:
                matches.append({
                    'normalized': disease,
                    'matches': match_count,
                    'total_symptoms': len(data['symptoms']),
                    'case_count': data['count'],
                    'original_names': data['original_names']
                })
        
        # Sort by match quality
        matches.sort(key=lambda x: (-x['matches'], -x['matches']/x['total_symptoms']))
        return matches[:5]

# Initialize analyzer
diagnosis_engine = SymptomAnalyzer()

async def start_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "🩺 Medical Symptom Analyzer\n"
        "Enter symptoms separated by commas (e.g., 'headache, fever')"
    )

async def analyze_symptoms(update: Update, context: CallbackContext) -> None:
    try:
        user_input = update.message.text.strip()
        if not user_input:
            await update.message.reply_text("Please enter symptoms.")
            return
            
        symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
        if not symptoms:
            await update.message.reply_text("No valid symptoms detected.")
            return
            
        results = diagnosis_engine.find_matches(symptoms)
        
        if not results:
            await update.message.reply_text("No matches found. Consult a doctor.")
            return
            
        response = ["🔍 Top Potential Diagnoses:"]
        for rank, match in enumerate(results, 1):
            # Get best original name
            original_name = next(iter(match['original_names']))
            
            # Get description
            desc_data = diagnosis_engine.descriptions.get(match['normalized'], {})
            description = desc_data.get('data', 'Description not available')
            
            # Get precautions
            prec_data = diagnosis_engine.precautions.get(match['normalized'], {})
            precautions = ', '.join(prec_data.get('data', ['Consult specialist']))
            
            response.append(
                f"{rank}. {original_name}\n"
                f"   📊 Matched {match['matches']} of {match['total_symptoms']} symptoms\n"
                f"   📈 Found in {match['case_count']} case variations\n"
                f"   📝 {description}\n"
                f"   ⚕️ Recommended: {precautions}"
            )
            
        await update.message.reply_text("\n\n".join(response))
        
    except Exception as error:
        await update.message.reply_text("⚠️ Error processing request. Try again.")
        logging.error(f"Error: {str(error)}")

def main():
    # Windows event loop fix
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Initialize bot
    bot_token = "yourtoken"
    app = Application.builder().token(bot_token).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symptoms))
    
    print("🟢 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🔴 Bot stopped")

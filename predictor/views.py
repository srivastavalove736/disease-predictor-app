import os
import pickle
import json
import numpy as np
import pypdf
from PIL import Image
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from xhtml2pdf import pisa
from .models import PredictionRecord

BASE_DIR = settings.BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, 'predictor', 'models_ml')

_models = {
    'db_model': None, 'db_scaler': None,
    'ht_model': None, 'ht_scaler': None,
    'skin_model': None, 'pk_model': None,
    'pk_scaler': None, 'brain_model': None,
}

SKIN_CLASSES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Nevi', 'Vascular Lesion']
BRAIN_TUMOR_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


# --- HELPER RESOURCE LOADERS ---
def load_diabetes_resources():
    if _models['db_model'] is None or _models['db_scaler'] is None:
        _models['db_model'] = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_model.pkl'), 'rb'))
        _models['db_scaler'] = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'), 'rb'))
    return _models['db_model'], _models['db_scaler']

def load_heart_resources():
    if _models['ht_model'] is None or _models['ht_scaler'] is None:
        _models['ht_model'] = pickle.load(open(os.path.join(MODEL_DIR, 'heart_model.pkl'), 'rb'))
        _models['ht_scaler'] = pickle.load(open(os.path.join(MODEL_DIR, 'heart_scaler.pkl'), 'rb'))
    return _models['ht_model'], _models['ht_scaler']

def load_skin_model():
    if _models['skin_model'] is None:
        import tensorflow as tf
        _models['skin_model'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'skin_cancer_resnet50.h5'))
    return _models['skin_model']

def load_parkinsons_resources():
    if _models['pk_model'] is None or _models['pk_scaler'] is None:
        _models['pk_model'] = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsons_model.pkl'), 'rb'))
        _models['pk_scaler'] = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsons_scaler.pkl'), 'rb'))
    return _models['pk_model'], _models['pk_scaler']

def load_brain_model():
    if _models['brain_model'] is None:
        import tensorflow as tf
        _models['brain_model'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'brain_tumor_resnet50.h5'))
    return _models['brain_model']


# --- HELPER: GENERATE AI INSIGHTS ---
def generate_ai_insights(disease_type, result):
    """
    Generates tailored actionable insights, precautions, and doctor questions based on the diagnosis.
    """
    insights = {
        'diet_lifestyle': [],
        'next_steps': [],
        'questions_for_doctor': []
    }

    # Parasitic / Malaria Handling
    if "Malaria" in result or "Plasmodium" in result or "Parasite" in result:
        insights['diet_lifestyle'] = [
            "Maintain strict hydration using clean water, electrolytes, and fluid soups.",
            "Get complete bed rest and ensure protection under insecticide-treated mosquito nets.",
            "Eat light, soft, nutritious meals rich in iron and protein to aid red blood cell recovery."
        ]
        insights['next_steps'] = [
            "Consult a physician or infectious disease expert immediately for prescription antimalarial therapy.",
            "Monitor platelet levels and CBC closely to evaluate fever or bleeding tendencies.",
            "Watch for warning signs like persistent vomiting, high fever with chills, or dark/tea-colored urine."
        ]
        insights['questions_for_doctor'] = [
            "Which specific antimalarial drug regimen is required based on the identified Plasmodium species?",
            "Will radical treatment (e.g., Primaquine) be needed to prevent dormant relapse?",
            "When should I repeat the blood smear or complete blood count (CBC) test?"
        ]

    # Hepatic / Liver / Jaundice / Hepatitis Handling
    elif "Hepatitis" in result or "Jaundice" in result or "Liver" in result or "Bilirubin" in result:
        insights['diet_lifestyle'] = [
            "Maintain high fluid intake, rest adequately, and strictly avoid alcohol or non-prescribed hepatotoxic drugs.",
            "Eat a bland, low-fat, highly digestible diet split into small, frequent meals.",
            "Avoid uncooked/raw seafood and maintain strict hand hygiene to prevent further infection transmission."
        ]
        insights['next_steps'] = [
            "Consult a Gastroenterologist or Hepatologist immediately for a liver function evaluation.",
            "Monitor for red-flag symptoms such as severe abdominal pain, persistent vomiting, confusion, or severe jaundice.",
            "Schedule follow-up LFTs and viral load testing as advised by your physician."
        ]
        insights['questions_for_doctor'] = [
            "Is this acute hepatitis profile caused by viral infection or other underlying liver stress?",
            "Which medications or supplements should I temporarily stop taking to reduce strain on my liver?",
            "How frequently should we monitor my Bilirubin, ALT/AST, and PT/INR values?"
        ]

    # Infectious / Typhoid Condition Handling
    elif "Infectious" in result or "Typhoid" in result or "Enteric" in result:
        insights['diet_lifestyle'] = [
            "Maintain strict oral and hand hygiene, drink only boiled/filtered water.",
            "Eat soft, high-calorie, easily digestible foods (e.g., rice gruel, soups, soft fruits).",
            "Ensure proper hydration using Oral Rehydration Solution (ORS) or electrolyte fluids."
        ]
        insights['next_steps'] = [
            "Consult a general physician or infectious disease specialist immediately.",
            "Complete the full antibiotic regimen as prescribed, even if symptoms clear up early.",
            "Log your daily body temperature to monitor fever trends."
        ]
        insights['questions_for_doctor'] = [
            "Which antibiotic treatment is recommended based on the blood culture sensitivity?",
            "Are there specific dietary precautions I should follow during my recovery?",
            "When should follow-up inflammatory tests (e.g., CRP or blood counts) be performed?"
        ]

    elif "Diabetes" in disease_type or "Hyperglycemia" in result:
        if "Positive" in result or "High Risk" in result or "Hyperglycemia" in result:
            insights['diet_lifestyle'] = [
                "Reduce intake of refined carbohydrates and simple sugars.",
                "Incorporate 30 minutes of moderate exercise daily (e.g., brisk walking).",
                "Monitor daily blood glucose levels using a home glucometer."
            ]
            insights['next_steps'] = [
                "Schedule an HbA1c blood test with an endocrinologist.",
                "Maintain an active food & blood sugar log to share with your physician."
            ]
            insights['questions_for_doctor'] = [
                "What target range should I aim for regarding my fasting blood sugar?",
                "Do I need preventive medication or can I manage this initially with diet?"
            ]
        else:
            insights['diet_lifestyle'] = [
                "Maintain a balanced diet rich in whole grains, fiber, and lean proteins.",
                "Stay regularly active to keep insulin sensitivity high."
            ]
            insights['next_steps'] = ["Continue regular annual health checkups."]
            insights['questions_for_doctor'] = ["How frequently should I repeat screening based on my risk factors?"]

    elif "Heart" in disease_type or "Cardiovascular" in result:
        if "High" in result:
            insights['diet_lifestyle'] = [
                "Adopt a Mediterranean or DASH diet (low sodium, healthy fats).",
                "Avoid smoking, excessive caffeine, and alcohol consumption.",
                "Practice daily stress-reduction techniques."
            ]
            insights['next_steps'] = [
                "Consult a cardiologist for a complete ECG / Echocardiogram evaluation.",
                "Check lipid panel profile (cholesterol, triglycerides)."
            ]
            insights['questions_for_doctor'] = [
                "Should I undergo a stress test or angiogram?",
                "What changes can I make immediately to lower my cardiovascular risk?"
            ]
        else:
            insights['diet_lifestyle'] = ["Maintain a heart-healthy active lifestyle with low sodium intake."]
            insights['next_steps'] = ["Schedule routine annual cardiac screenings."]
            insights['questions_for_doctor'] = ["What is my optimal blood pressure target range?"]

    elif "Anemia" in result or "Hemoglobin" in result:
        insights['diet_lifestyle'] = [
            "Increase intake of iron-rich foods (e.g., spinach, legumes, lean meat).",
            "Pair iron sources with Vitamin C to improve absorption."
        ]
        insights['next_steps'] = [
            "Consult a physician to determine the root cause of low hemoglobin.",
            "Get serum ferritin and iron profile levels evaluated."
        ]
        insights['questions_for_doctor'] = [
            "Do I require dietary iron supplements?",
            "What follow-up blood work is necessary to track my hemoglobin recovery?"
        ]

    elif "Skin" in disease_type:
        insights['diet_lifestyle'] = [
            "Apply broad-spectrum SPF 30+ sunscreen daily.",
            "Avoid direct peak sun exposure between 10 AM and 4 PM."
        ]
        insights['next_steps'] = [
            "Schedule a clinical dermoscopy with a certified dermatologist.",
            "Monitor the lesion for changes using the ABCDE rule."
        ]
        insights['questions_for_doctor'] = ["Does this lesion require a skin biopsy for full verification?"]

    elif "Brain" in disease_type:
        insights['diet_lifestyle'] = [
            "Ensure adequate rest and avoid sleep deprivation.",
            "Keep track of any neurological symptoms (headaches, nausea, vision changes)."
        ]
        insights['next_steps'] = [
            "Consult a neurologist or neuro-oncologist to evaluate the scan.",
            "Obtain a contrast-enhanced MRI if recommended by a specialist."
        ]
        insights['questions_for_doctor'] = [
            "What further imaging or tests are required to confirm this MRI finding?",
            "What symptoms should prompt an emergency room visit?"
        ]

    elif "Report" in disease_type or "PDF" in disease_type:
        insights['diet_lifestyle'] = [
            "Review the extracted report flags and discuss specific findings with your doctor.",
            "Maintain proper hydration and ensure regular sleep schedules."
        ]
        insights['next_steps'] = [
            "Share the full original laboratory report with your primary physician.",
            "Schedule follow-up lab work for flagged abnormal values."
        ]
        insights['questions_for_doctor'] = [
            "Are any values flagged in this report critical or urgent?",
            "Should we order follow-up laboratory testing to confirm these results?"
        ]

    else:  # Parkinson's
        insights['diet_lifestyle'] = [
            "Engage in balance, flexibility, and physical therapy exercises.",
            "Consume a high-fiber diet to support overall nervous system & gut health."
        ]
        insights['next_steps'] = [
            "Consult a movement disorder specialist or neurologist.",
            "Undergo a comprehensive clinical motor skill assessment."
        ]
        insights['questions_for_doctor'] = ["What early management strategies or therapies are recommended?"]

    return insights


# --- AUTHENTICATION VIEWS ---
def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')


# --- MAIN PREDICTION VIEWS ---
@login_required(login_url='login')
def home(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def predict_diabetes(request):
    if request.method == 'POST':
        db_model, db_scaler = load_diabetes_resources()
        raw_inputs = [float(x) for x in request.POST.getlist('features')]
        scaled_features = db_scaler.transform([raw_inputs])
        pred = db_model.predict(scaled_features)[0]

        result = "Positive for Diabetes" if pred == 1 else "Negative for Diabetes"
        rec = PredictionRecord.objects.create(user=request.user, disease_type='Diabetes', result=result)
        ai_insights = generate_ai_insights('Diabetes', result)

        return render(request, 'result.html', {
            'prediction': result,
            'title': 'Diabetes Assessment',
            'disease_type': 'Diabetes',
            'record_id': rec.id,
            'ai_insights': ai_insights
        })
    return redirect('home')

@login_required(login_url='login')
def predict_heart(request):
    if request.method == 'POST':
        ht_model, ht_scaler = load_heart_resources()
        raw_inputs = [float(x) for x in request.POST.getlist('features')]
        scaled_features = ht_scaler.transform([raw_inputs])
        pred = ht_model.predict(scaled_features)[0]

        result = "High Risk of Heart Disease" if pred == 1 else "Low Risk of Heart Disease"
        rec = PredictionRecord.objects.create(user=request.user, disease_type='Heart', result=result)
        ai_insights = generate_ai_insights('Heart', result)

        return render(request, 'result.html', {
            'prediction': result,
            'title': 'Heart Disease Risk',
            'disease_type': 'Heart',
            'record_id': rec.id,
            'ai_insights': ai_insights
        })
    return redirect('home')

@login_required(login_url='login')
def predict_skin(request):
    if request.method == 'POST' and request.FILES.get('skin_image'):
        from tensorflow.keras.applications.resnet50 import preprocess_input
        skin_model = load_skin_model()
        image_file = request.FILES['skin_image']

        img = Image.open(image_file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = skin_model.predict(img_array)
        
        # Check if output is raw logits or softmax probabilities
        if np.isclose(np.sum(preds[0]), 1.0, atol=1e-2):
            probs = preds[0]
        else:
            exp_p = np.exp(preds[0] - np.max(preds[0]))
            probs = exp_p / exp_p.sum()

        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx] * 100)

        result = f"Diagnosis: {SKIN_CLASSES[class_idx]}"
        
        rec = PredictionRecord.objects.create(
            user=request.user, 
            disease_type='Skin', 
            result=result, 
            confidence=confidence,
            image=image_file
        )
        ai_insights = generate_ai_insights('Skin', result)

        return render(request, 'result.html', {
            'prediction': result,
            'image_url': rec.image.url if rec.image else None,
            'title': 'Skin Cancer Screening',
            'disease_type': 'Skin',
            'confidence': confidence,
            'record_id': rec.id,
            'ai_insights': ai_insights
        })
    return redirect('home')

@login_required(login_url='login')
def predict_parkinsons(request):
    if request.method == 'POST':
        pk_model, pk_scaler = load_parkinsons_resources()
        raw_inputs = [float(x) for x in request.POST.getlist('features')]
        scaled_features = pk_scaler.transform([raw_inputs])
        pred = pk_model.predict(scaled_features)[0]
        
        result = "Positive for Parkinson's Indicators" if pred == 1 else "Negative for Parkinson's"
        rec = PredictionRecord.objects.create(user=request.user, disease_type='Parkinsons', result=result)
        ai_insights = generate_ai_insights('Parkinsons', result)

        return render(request, 'result.html', {
            'prediction': result,
            'title': 'Parkinsons Risk Assessment',
            'disease_type': 'Parkinsons',
            'record_id': rec.id,
            'ai_insights': ai_insights
        })
    return redirect('home')

@login_required(login_url='login')
def predict_brain(request):
    if request.method == 'POST' and request.FILES.get('mri_image'):
        from tensorflow.keras.applications.resnet50 import preprocess_input
        brain_model = load_brain_model()
        image_file = request.FILES['mri_image']

        img = Image.open(image_file).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = brain_model.predict(img_array)

        # Check if output is raw logits or softmax probabilities
        if np.isclose(np.sum(preds[0]), 1.0, atol=1e-2):
            probs = preds[0]
        else:
            exp_p = np.exp(preds[0] - np.max(preds[0]))
            probs = exp_p / exp_p.sum()

        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx] * 100)

        result = f"Diagnosis: {BRAIN_TUMOR_CLASSES[class_idx]}"
        
        rec = PredictionRecord.objects.create(
            user=request.user, 
            disease_type='Brain Tumor', 
            result=result, 
            confidence=confidence,
            image=image_file
        )
        ai_insights = generate_ai_insights('Brain Tumor', result)

        return render(request, 'result.html', {
            'prediction': result,
            'image_url': rec.image.url if rec.image else None,
            'title': 'Brain Tumor MRI Analysis',
            'disease_type': 'Brain Tumor',
            'confidence': confidence,
            'record_id': rec.id,
            'ai_insights': ai_insights
        })
    return redirect('home')

@login_required(login_url='login')
def predict_report(request):
    if request.method == 'POST' and request.FILES.get('report_pdf'):
        pdf_file = request.FILES['report_pdf']
        
        # 1. Parse text out of PDF
        extracted_text = ""
        try:
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text.lower() + " "
        except Exception:
            return render(request, 'index.html', {'error': 'Failed to process PDF report file.'})

        # 2. Refined Rule-Based NLP Keyword Extraction
        result = "Normal / No High Risk Indicators Detected"

        # 1. Check for Malaria & Parasitic Vector-Borne Markers FIRST
        malaria_keywords = ['malaria', 'plasmodium', 'pldh', 'hrp-2', 'vivax', 'falciparum', 'parasitemia', 'trophozoites', 'ring forms']
        if any(kw in extracted_text for kw in malaria_keywords):
            if any(term in extracted_text for term in ['positive', 'detected', 'seen', 'high', 'reactive']):
                result = "Infectious Indication: Active Malaria Parasite Infection Detected"
            else:
                result = "Parasite Screening: Malaria Markers Non-Reactive"

        # 2. Check for Typhoid / Enteric Fever Markers SECOND
        elif any(kw in extracted_text for kw in ['typhoid', 'salmonella', 'widal', 'typhidot', 'enteric fever', 's. typhi']):
            if any(term in extracted_text for term in ['positive', 'reactive', 'isolated', '1:160', '1:320', 'high']):
                result = "Infectious Indication: Active Enteric Fever / Typhoid Detected"
            else:
                result = "Serology Evaluation: Typhoid Markers Present (Non-Reactive / Low Titer)"

        # 3. Check for Liver / Hepatitis / Jaundice Markers
        elif any(kw in extracted_text for kw in ['jaundice', 'hepatitis', 'hepatobiliary', 'hav', 'hbv', 'hcv']):
            if any(term in extracted_text for term in ['high', 'elevated', 'reactive', 'positive', 'abnormal']):
                result = "Hepatic Indication: Acute Viral Hepatitis / Jaundice Detected"
            else:
                result = "Liver Profile: Normal Range"

        # 4. Check for Cardiovascular / Lipid Profile
        elif any(kw in extracted_text for kw in ['cholesterol', 'triglycerides', 'troponin', 'lipid']):
            if any(term in extracted_text for term in ['high', 'abnormal', 'elevated', 'borderline']):
                result = "High Risk: Cardiovascular / Lipid Anomaly"
            else:
                result = "Normal Range: Lipid & Cardiac Markers"

        # 5. Check for Glycemic / Diabetes Indicators
        elif 'hba1c' in extracted_text or ('glucose' in extracted_text and any(term in extracted_text for term in ['hyperglycemia', 'diabetic', 'prediabetes'])):
            if any(term in extracted_text for term in ['high', 'elevated']):
                result = "High Risk: Hyperglycemia / Diabetes Indication"
            else:
                result = "Normal Range: Glycemic Control"

        # 6. Check for Anemia / Hematology Profile LAST
        elif any(kw in extracted_text for kw in ['hemoglobin', 'rbc', 'iron', 'ferritin']):
            if any(term in extracted_text for term in ['low', 'anemia', 'deficiency']):
                result = "Risk Identified: Low Hemoglobin / Anemia"

        # 3. Save to database
        rec = PredictionRecord.objects.create(user=request.user, disease_type='PDF Report Analysis', result=result)
        ai_insights = generate_ai_insights('PDF Report', result)

        return render(request, 'result.html', {
            'prediction': result,
            'title': 'Patient PDF Medical Report Analysis',
            'disease_type': 'PDF Report Analysis',
            'record_id': rec.id,
            'ai_insights': ai_insights,
            'extracted_text_snippet': extracted_text[:300] + "..." if extracted_text else "No text extracted."
        })

    return redirect('home')


# --- AI CHATBOT API ENDPOINT ---
@login_required(login_url='login')
def ai_chat_api(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        disease_type = data.get('disease_type', '')
        result = data.get('result', '')

        if not user_message:
            return JsonResponse({'reply': 'Please enter a valid question.'})

        msg_lower = user_message.lower()

        if 'doctor' in msg_lower or 'specialist' in msg_lower:
            reply = f"For {disease_type} showing '{result}', it is strongly advised to consult a primary care physician or relevant medical specialist for formal clinical evaluation."
        elif 'diet' in msg_lower or 'food' in msg_lower or 'eat' in msg_lower:
            reply = f"A balanced diet tailored to {disease_type} care is key. Focus on whole foods, proper hydration, and limiting processed or high-sugar/salty foods."
        elif 'cure' in msg_lower or 'treat' in msg_lower:
            reply = f"Treatment strategies depend on official diagnostic confirmation by a physician, who may recommend lifestyle changes, therapies, or medication."
        elif 'accurate' in msg_lower or 'reliable' in msg_lower or 'ai' in msg_lower:
            reply = "This assessment is generated by a Machine Learning model for preliminary screening only. It does not replace professional medical diagnosis."
        else:
            reply = f"Regarding '{user_message}': In the context of {disease_type} screening ({result}), monitor any symptoms closely and discuss them during your next medical appointment."

        return JsonResponse({'reply': reply})
    return JsonResponse({'error': 'Invalid request method'}, status=400)


# --- HISTORY & ANALYTICS VIEW ---
@login_required(login_url='login')
def history(request):
    records = PredictionRecord.objects.filter(user=request.user).order_by('-created_at')
    
    disease_counts = list(PredictionRecord.objects.filter(user=request.user)
                          .values('disease_type')
                          .annotate(count=Count('id')))
    
    labels = [item['disease_type'] for item in disease_counts]
    counts = [item['count'] for item in disease_counts]

    return render(request, 'history.html', {
        'records': records,
        'chart_labels': labels,
        'chart_counts': counts,
    })


# --- PDF GENERATION VIEW ---
@login_required(login_url='login')
def download_pdf(request, record_id):
    record = get_object_or_404(PredictionRecord, id=record_id, user=request.user)
    
    # Generate the AI insights dynamically for the PDF render context
    ai_insights = generate_ai_insights(record.disease_type, record.result)
    
    template_path = 'pdf_template.html'
    context = {
        'record': record,
        'user': request.user,
        'ai_insights': ai_insights
    }

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="Diagnostic_Report_{record.id}.pdf"'

    template = render(request, template_path, context)
    pisa_status = pisa.CreatePDF(template.content.decode('utf-8'), dest=response)
    
    if pisa_status.err:
        return HttpResponse('Failed to generate PDF report', status=500)
    return response
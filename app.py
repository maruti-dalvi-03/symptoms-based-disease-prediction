from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from disease_info import disease_info

app = Flask(__name__)

# Load the trained Naive Bayes model
# model = joblib.load('naive_bayes_model.joblib')

# Load trained models
loaded_gnb = joblib.load('./models/naive_bayes_model.joblib')
loaded_dtc = joblib.load('./models/decision_tree_model.joblib')
loaded_rfc = joblib.load('./models/random_forest_model.joblib')
loaded_knn = joblib.load('./models/knn_model.joblib')


# List of symptoms and diseases (same as in your previous code)
# List of symptoms
l1 = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
      'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
      'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
      'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
      'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
      'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
      'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
      'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
      'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
      'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
      'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
      'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases',
      'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
      'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
      'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
      'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
      'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
      'red_sore_around_nose', 'yellow_crust_ooze']


subcategories = {
    'skin_related_symptoms': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'ulcers_on_tongue', 'yellowish_skin', 'dark_urine',
                     'red_spots_over_body', 'belly_pain', 'dischromic _patches', 'watering_from_eyes', 'pus_filled_pimples',
                     'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
                     'blister', 'red_sore_around_nose', 'yellow_crust_ooze'],
    'cold_related_symptoms': ['continuous_sneezing', 'shivering', 'chills', 'cold_hands_and_feets'],
    'pain_related_symptoms': ['joint_pain', 'muscle_wasting', 'pain_behind_the_eyes', 'back_pain', 'neck_pain', 'dizziness', 'cramps',
                     'bruising', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'painful_walking'],
    'stomach_and_abdomen_related_symptoms': ['stomach_pain', 'acidity', 'vomiting', 'burning_micturition', 'indigestion', 'abdominal_pain',
                                    'constipation', 'diarrhoea', 'swelling_of_stomach', 'malaise', 'distention_of_abdomen',
                                    'stomach_bleeding'],
    'face_and_throat_related_symptoms': ['mood_swings', 'patches_in_throat', 'cough', 'headache', 'sunken_eyes', 'breathlessness',
                                'dehydration', 'yellowing_of_eyes', 'blurred_and_distorted_vision', 'redness_of_eyes',
                                'runny_nose', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'neck_pain'],
    'heart_and_muscles_related_symptoms': ['fatigue', 'weight_gain', 'anxiety', 'weight_loss', 'restlessness',
                                  'lethargy', 'high_fever', 'muscle_pain', 'pain_during_bowel_movements', 'pain_in_anal_region',
                                  'bloody_stool', 'irritation_in_anus', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
                                  'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
                                  'excessive_hunger', 'extra_marital_contacts', 'slurred_speech', 'movement_stiffness',
                                  'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
                                  'loss_of_smell', 'palpitations', 'pus_filled_pimples'],
    'general_symptoms': ['anxiety', 'restlessness', 'lethargy', 'malaise', 'altered_sensorium', 'receiving_blood_transfusion',
                'receiving_unsterile_injections', 'coma', 'history_of_alcohol_consumption'],
    'urine_related_symptoms': ['spotting_urination', 'yellow_urine', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases',
                      'internal_itching', 'toxic_look_(typhos)', 'polyuria'],
    'mental_issue_related_symptoms': ['anxiety', 'mood_swings', 'restlessness', 'depression', 'irritability', 'lack_of_concentration']
}

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
           'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
           'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemorrhoids(piles)',
           'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthritis',
           'Arthritis', '(vertigo) Paroxysmal Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']


@app.route('/')
def home():
    return render_template('index.html', symptoms=l1 ,size=len(l1))

@app.route('/about')
def about():
    return render_template('about.html', symptoms=l1)

@app.route('/symptons-based-prediction')
def predictions():
    return render_template('symptons-based-prediction.html', subcategories=subcategories)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.getlist('symptoms')

    input_test = pd.DataFrame(0, index=[0], columns=l1)
    for symptom in symptoms:
        input_test[symptom] = 1

    # Predict using Naive Bayes
    predict_naive_bayes = loaded_gnb.predict(input_test)
    predict_proba_naive_bayes = loaded_gnb.predict_proba(input_test)

    # Predict using Decision Tree
    predict_decision_tree = loaded_dtc.predict(input_test)
    predict_proba_decision_tree = loaded_dtc.predict_proba(input_test)

    # Predict using Random Forest
    predict_random_forest = loaded_rfc.predict(input_test)
    predict_proba_random_forest = loaded_rfc.predict_proba(input_test)

    # Predict using K-Nearest Neighbors (KNN)
    predict_knn = loaded_knn.predict(input_test)
    predict_proba_knn = loaded_knn.predict_proba(input_test)

    # Get the predicted disease name
    predicted_disease_naive_bayes = disease[predict_naive_bayes[0]]
    predicted_disease_decision_tree = disease[predict_decision_tree[0]]
    predicted_disease_random_forest = disease[predict_random_forest[0]]
    predicted_disease_knn = disease[predict_knn[0]]

    # Get the probability for the predicted disease
    probability_naive_bayes = max(predict_proba_naive_bayes[0])
    probability_decision_tree = max(predict_proba_decision_tree[0])
    probability_random_forest = max(predict_proba_random_forest[0])
    probability_knn = max(predict_proba_knn[0])

    return render_template('result.html',
                           symptoms=symptoms,
                           disease_naive_bayes=predicted_disease_naive_bayes,
                           probability_naive_bayes=probability_naive_bayes * 100,
                           disease_decision_tree=predicted_disease_decision_tree,
                           probability_decision_tree=probability_decision_tree * 100,
                           disease_random_forest=predicted_disease_random_forest,
                           probability_random_forest=probability_random_forest * 100,
                           disease_knn=predicted_disease_knn,
                           probability_knn=probability_knn * 100)

@app.route('/all-services')
def allServices():
    return render_template('allServices.html')

@app.route('/all-services/diabetes')
def diabetesPredict():
    # # Get user input from form
    # input_features = [float(x) for x in request.form.values()]
    
    # # Convert input to numpy array and reshape
    # input_array = np.array(input_features).reshape(1, -1)
    
    # # Make prediction
    # prediction = model.predict(input_array)
    # prediction_proba = model.predict_proba(input_array)
    
    # # Map predicted class to label
    # class_labels = {
    #     0: "No diabetes or only during pregnancy",
    #     1: "Prediabetes",
    #     2: "Diabetes"
    # }
    # predicted_label = class_labels.get(prediction[0], "Unknown")

    # Return prediction result
    # return render_template('diabetes.html', prediction=predicted_label, prediction_proba=prediction_proba)
    return render_template('diabetes.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

# @app.template_filter('capitalize_words')
# def capitalize_words(s):
#     return ' '.join(word.capitalize() for word in s.split('_'))

if __name__ == '__main__':
    app.run(debug=True)

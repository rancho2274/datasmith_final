import json
from datetime import datetime, timedelta
import random

def create_dummy_data(num_patients=35):
    """Generates a diverse dictionary of dummy patient discharge reports."""
    
    # Expanded list of names
    first_names = [
        "John", "Jane", "Michael", "Emily", "Robert", "Maria", "David", "Jessica",
        "Chris", "Sarah", "James", "Linda", "Thomas", "Nancy", "Daniel", "Laura",
        "Kevin", "Lisa", "Paul", "Betty", "William", "Karen", "Steven", "Anna",
        "Mark", "Patricia", "Jennifer", "Richard", "Susan", "Joseph", "Margaret",
        "Charles", "Elizabeth", "Matthew", "Ashley", "Andrew", "Michelle", "Joshua"
    ]
    
    last_names = [
        "Smith", "Doe", "Johnson", "Williams", "Brown", "Garcia", "Miller", "Davis",
        "Wilson", "Martinez", "Lee", "Chen", "White", "Kim", "Harris", "Lopez",
        "Walker", "Hall", "Young", "Allen", "King", "Wright", "Scott", "Adams",
        "Baker", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner"
    ]
    
    # Expanded diagnoses with stages and variations
    diagnoses_data = [
        {
            "diagnosis": "Chronic Kidney Disease Stage 3a",
            "medications": ["Lisinopril 10mg daily", "Metformin 500mg twice daily"],
            "dietary": "Low sodium (2g/day), moderate protein (0.8g/kg/day)",
            "warning_signs": "Fatigue, swelling in ankles, changes in urination",
            "instructions": "Monitor blood pressure daily, check blood sugar if diabetic, follow-up in 3 months"
        },
        {
            "diagnosis": "Chronic Kidney Disease Stage 3b",
            "medications": ["Losartan 50mg daily", "Furosemide 20mg daily", "Calcium carbonate 500mg with meals"],
            "dietary": "Low sodium (2g/day), low phosphorus, moderate protein",
            "warning_signs": "Swelling, shortness of breath, decreased urine output, bone pain",
            "instructions": "Monitor blood pressure, take phosphate binders with meals, weigh daily"
        },
        {
            "diagnosis": "Chronic Kidney Disease Stage 4",
            "medications": ["Lisinopril 10mg daily", "Furosemide 40mg twice daily", "Sevelamer 800mg with meals", "Erythropoietin injection weekly"],
            "dietary": "Low sodium (2g/day), low potassium, low phosphorus, fluid restriction (1.5L/day)",
            "warning_signs": "Severe swelling, shortness of breath, decreased urine output, extreme fatigue, nausea",
            "instructions": "Monitor blood pressure daily, weigh yourself daily, prepare for dialysis evaluation, take all medications with meals"
        },
        {
            "diagnosis": "Chronic Kidney Disease Stage 5 (ESRD)",
            "medications": ["Amlodipine 5mg daily", "Furosemide 40mg daily", "Sevelamer 1600mg with meals", "Erythropoietin injection twice weekly"],
            "dietary": "Strict fluid restriction (1L/day), low sodium (1.5g/day), low potassium, low phosphorus",
            "warning_signs": "Severe shortness of breath, chest pain, confusion, no urine output, severe swelling",
            "instructions": "Dialysis scheduled, strict fluid and diet compliance, weigh before and after dialysis, emergency contact ready"
        },
        {
            "diagnosis": "Acute Kidney Injury (AKI) - Prerenal",
            "medications": ["IV fluids completed", "Furosemide 20mg twice daily", "Monitor electrolytes"],
            "dietary": "Low sodium (2g/day), adequate hydration (2L/day unless restricted)",
            "warning_signs": "Decreased urine output, dizziness, rapid weight loss, dry mouth",
            "instructions": "Monitor urine output, stay hydrated, follow-up in 1 week, repeat labs in 3 days"
        },
        {
            "diagnosis": "Acute Kidney Injury (AKI) - Intrinsic",
            "medications": ["Prednisone 40mg daily (tapering)", "Furosemide 40mg daily", "Sodium bicarbonate 650mg twice daily"],
            "dietary": "Low sodium (2g/day), low potassium, fluid restriction (1.5L/day)",
            "warning_signs": "Decreased urine output, swelling, shortness of breath, dark urine",
            "instructions": "Monitor urine output closely, take steroids as prescribed, follow-up in 1 week, urgent care if no urine for 12 hours"
        },
        {
            "diagnosis": "Hypertensive Nephropathy",
            "medications": ["Lisinopril 20mg daily", "Amlodipine 10mg daily", "Hydrochlorothiazide 25mg daily"],
            "dietary": "Strict low sodium (1.5g/day), DASH diet recommended, limit alcohol",
            "warning_signs": "High blood pressure readings, headaches, vision changes, chest pain",
            "instructions": "Monitor blood pressure twice daily, take medications at same time daily, follow-up in 2 weeks, lifestyle modifications"
        },
        {
            "diagnosis": "Diabetic Nephropathy",
            "medications": ["Lisinopril 10mg daily", "Metformin 1000mg twice daily", "Glipizide 5mg daily", "Atorvastatin 20mg daily"],
            "dietary": "Diabetic diet, low sodium (2g/day), carbohydrate counting, limit processed foods",
            "warning_signs": "High blood sugar, protein in urine, swelling, vision changes, foot numbness",
            "instructions": "Monitor blood sugar 4 times daily, check feet daily, A1C goal <7%, follow-up in 1 month"
        },
        {
            "diagnosis": "Polycystic Kidney Disease (PKD)",
            "medications": ["Tolvaptan 45mg twice daily", "Lisinopril 10mg daily", "Pain management as needed"],
            "dietary": "Low sodium (2g/day), adequate hydration (2.5L/day), avoid caffeine",
            "warning_signs": "Severe back/flank pain, blood in urine, high blood pressure, frequent UTIs",
            "instructions": "Monitor blood pressure, stay well-hydrated, avoid contact sports, genetic counseling recommended, follow-up in 3 months"
        },
        {
            "diagnosis": "Glomerulonephritis",
            "medications": ["Prednisone 60mg daily (tapering)", "Mycophenolate 500mg twice daily", "Lisinopril 10mg daily"],
            "dietary": "Low sodium (2g/day), low protein during active phase, fluid restriction if edematous",
            "warning_signs": "Blood in urine, foamy urine, swelling, high blood pressure, fatigue",
            "instructions": "Take immunosuppressants exactly as prescribed, monitor for infections, follow-up in 2 weeks, urgent care if fever"
        },
        {
            "diagnosis": "Nephrotic Syndrome",
            "medications": ["Prednisone 1mg/kg daily", "Furosemide 40mg daily", "Atorvastatin 20mg daily", "Warfarin 5mg daily"],
            "dietary": "Low sodium (1.5g/day), low fat, moderate protein, fluid restriction (1L/day)",
            "warning_signs": "Severe swelling (especially face and legs), foamy urine, weight gain, blood clots",
            "instructions": "Weigh daily, monitor for blood clots, take anticoagulant as prescribed, follow-up in 1 week"
        },
        {
            "diagnosis": "Kidney Stones (Nephrolithiasis)",
            "medications": ["Tamsulosin 0.4mg daily", "Ibuprofen 600mg as needed for pain", "Potassium citrate 10mEq twice daily"],
            "dietary": "High fluid intake (3L/day), low oxalate, limit sodium, moderate calcium",
            "warning_signs": "Severe flank pain, blood in urine, fever, inability to urinate, nausea/vomiting",
            "instructions": "Drink plenty of water, strain urine for stones, follow-up in 2 weeks, emergency if unable to urinate"
        },
        {
            "diagnosis": "Urinary Tract Infection with Kidney Involvement",
            "medications": ["Ciprofloxacin 500mg twice daily for 7 days", "Phenazopyridine 200mg three times daily as needed"],
            "dietary": "Increase fluid intake (2.5L/day), cranberry juice recommended, avoid irritants",
            "warning_signs": "Fever, chills, back pain, burning urination, blood in urine, confusion (elderly)",
            "instructions": "Complete full course of antibiotics, increase fluids, follow-up if symptoms persist, urgent care if fever >101F"
        },
        {
            "diagnosis": "Kidney Transplant - Post-Operative",
            "medications": ["Tacrolimus 3mg twice daily", "Mycophenolate 1000mg twice daily", "Prednisone 20mg daily (tapering)", "Valganciclovir 450mg daily"],
            "dietary": "Low sodium (2g/day), well-cooked foods only, avoid raw foods, adequate protein",
            "warning_signs": "Fever, decreased urine output, pain at transplant site, signs of rejection, infection",
            "instructions": "Take immunosuppressants exactly on time, monitor for infections, avoid crowds, follow-up in 1 week, emergency contact for rejection signs"
        },
        {
            "diagnosis": "IgA Nephropathy",
            "medications": ["Lisinopril 10mg daily", "Fish oil 2g twice daily", "Prednisone 40mg daily if indicated"],
            "dietary": "Low sodium (2g/day), moderate protein, heart-healthy diet",
            "warning_signs": "Blood in urine (especially after infections), proteinuria, high blood pressure, swelling",
            "instructions": "Monitor blood pressure, take fish oil supplements, follow-up in 3 months, avoid NSAIDs"
        }
    ]
    
    # Follow-up variations
    follow_up_options = [
        "Nephrology clinic in 1 week",
        "Nephrology clinic in 2 weeks",
        "Nephrology clinic in 1 month",
        "Nephrology clinic in 3 months",
        "Primary care in 2 weeks, Nephrology in 1 month",
        "Urgent follow-up in 3 days",
        "Dialysis center evaluation in 1 week",
        "Transplant clinic in 2 weeks"
    ]
    
    patient_data = {}
    used_names = set()
    
    for i in range(num_patients):
        # Generate unique name
        while True:
            first = random.choice(first_names)
            last = random.choice(last_names)
            name = f"{first} {last}"
            if name not in used_names:
                used_names.add(name)
                break
        
        # Select diagnosis data
        diag_data = random.choice(diagnoses_data)
        
        # Create report with variations
        report = {
            "discharge_date": (datetime.now() - timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d"),
            "primary_diagnosis": diag_data["diagnosis"],
            "medications": diag_data["medications"].copy(),
            "dietary_restrictions": diag_data["dietary"],
            "follow_up": random.choice(follow_up_options),
            "warning_signs": diag_data["warning_signs"],
            "discharge_instructions": diag_data["instructions"],
            "age": random.randint(25, 85),
            "gender": random.choice(["Male", "Female", "Other"]),
            "admission_reason": random.choice([
                "Routine check-up",
                "Worsening kidney function",
                "Acute symptoms",
                "Medication adjustment",
                "Dialysis initiation",
                "Post-surgical care",
                "Infection management"
            ])
        }
        
        # Add lab values for some patients
        if random.random() > 0.3:  # 70% of patients have lab values
            report["recent_labs"] = {
                "creatinine": round(random.uniform(1.2, 8.5), 2),
                "eGFR": random.randint(8, 60),
                "potassium": round(random.uniform(3.5, 6.2), 1),
                "sodium": random.randint(135, 145)
            }
        
        patient_data[name] = report
    
    return patient_data

# Generate and save the data
if __name__ == "__main__":
    DUMMY_PATIENT_DATA = create_dummy_data(35)
    with open("patient_data.json", "w") as f:
        json.dump(DUMMY_PATIENT_DATA, f, indent=4)
    
    print(f"patient_data.json created with {len(DUMMY_PATIENT_DATA)} records.")
    print(f"Diagnoses included: {set([p['primary_diagnosis'] for p in DUMMY_PATIENT_DATA.values()])}")

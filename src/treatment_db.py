# src/treatment_db.py 
# ───────────────────────────────────────────────────────────────── 
# Rule-based treatment recommendation database. 
# In production, this would be curated by agronomists. 
# ───────────────────────────────────────────────────────────────── 

TREATMENT_DB = { 
    "Potato___Early_blight": { 
        "pathogen": "Alternaria solani (fungal)", 
        "mild": { 
            "pesticide":  "Mancozeb 75WP @ 2g/litre water", 
            "frequency":  "Every 10 days; cease 3 weeks before harvest", 
            "fertilizer": "Increase potassium (K₂O @ 60 kg/ha)", 
            "action":     "Remove infected lower leaves; avoid wet foliage" 
        }, 
        "severe": { 
            "pesticide":  "Chlorothalonil 75WP @ 2g/litre OR Azoxystrobin 23SC @ 1ml/litre", 
            "frequency":  "Every 7 days", 
            "fertilizer": "Reduce nitrogen; apply calcium+boron foliar spray", 
            "action":     "Isolate field; urgent KVK consultation recommended" 
        } 
    }, 
    "Potato___Late_blight": { 
        "pathogen": "Phytophthora infestans (oomycete)", 
        "mild": { 
            "pesticide":  "Cymoxanil + Mancozeb @ 3g/litre", 
            "frequency":  "Every 7 days", 
            "fertilizer": "Balanced NPK; avoid excess nitrogen", 
            "action":     "Destroy crop debris; ensure field drainage" 
        }, 
        "severe": { 
            "pesticide":  "Metalaxyl-M + Mancozeb @ 2.5g/litre", 
            "frequency":  "Every 5–7 days in high humidity", 
            "fertilizer": "Potassium-rich: MOP @ 80 kg/ha", 
            "action":     "Seriously consider early harvest to salvage tubers" 
        } 
    }, 
    "Tomato___Early_blight": { 
        "pathogen": "Alternaria solani (fungal)", 
        "mild": { 
            "pesticide":  "Mancozeb 75WP @ 2g/litre", 
            "frequency":  "Every 10 days", 
            "fertilizer": "Balanced NPK with micronutrients", 
            "action":     "Prune affected leaves; improve spacing for airflow" 
        }, 
        "severe": { 
            "pesticide":  "Propiconazole 25EC @ 1ml/litre", 
            "frequency":  "Every 7 days; 3-spray rotation", 
            "fertilizer": "Reduce nitrogen; supplement potassium + calcium", 
            "action":     "Quarantine section; remove and burn debris" 
        } 
    }, 
    "Tomato___Late_blight": { 
        "pathogen": "Phytophthora infestans (oomycete)", 
        "mild": { 
            "pesticide":  "Copper oxychloride 50WP @ 3g/litre", 
            "frequency":  "Weekly; stop 2 weeks before harvest", 
            "fertilizer": "Normal NPK; avoid overhead irrigation", 
            "action":     "Scout daily; mark infected plants" 
        }, 
        "severe": { 
            "pesticide":  "Dimethomorph + Mancozeb @ 2g/litre", 
            "frequency":  "Every 5 days in wet weather", 
            "fertilizer": "Potassium supplement; reduce leaf wetness", 
            "action":     "Emergency KVK consultation" 
        } 
    }, 
    "Apple___Apple_scab": { 
        "pathogen": "Venturia inaequalis (fungal)", 
        "mild": { 
            "pesticide":  "Captan 50WP @ 2g/litre", 
            "frequency":  "Pre-bloom + early post-bloom sprays", 
            "fertilizer": "Balanced NPK", 
            "action":     "Rake and destroy fallen leaves" 
        }, 
        "severe": { 
            "pesticide":  "Myclobutanil 10WP @ 0.5g/litre", 
            "frequency":  "Every 7 days during wet spring", 
            "fertilizer": "Reduce nitrogen; maintain pH 6.5–7.0", 
            "action":     "Prune for airflow; use resistant varieties next season" 
        } 
    }, 
} 

REGION_NAMES   = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Central"} 
KVK_HELPLINE   = "1800-180-1551 (Kisan Call Centre, toll-free)" 

DEFAULT_TREATMENT = { 
    "pesticide":  "Consult local agricultural office or KVK", 
    "frequency":  "N/A", 
    "fertilizer": "Maintain balanced NPK", 
    "action":     "Submit leaf sample for laboratory confirmation" 
} 

def get_recommendation(disease_class: str, 
                        confidence: float, 
                        region: int = 0) -> dict: 
    """ 
    Returns treatment recommendation for a given disease and severity. 

    Args: 
        disease_class: e.g. "Potato___Early_blight" 
        confidence:    CNN softmax confidence (0–1) 
        region:        0=North, 1=South, 2=East, 3=West, 4=Central 

    Returns: 
        dict with pesticide, frequency, fertilizer, action, region_note 
    """ 
    severity_label = "severe" if confidence >= 0.70 else "mild" 

    entry = TREATMENT_DB.get(disease_class, {}) 
    treatment = entry.get(severity_label, DEFAULT_TREATMENT).copy() 

    disease_readable = ( 
        disease_class 
        .replace("___", " — ") 
        .replace("_", " ") 
    ) 

    treatment["disease"]    = disease_readable 
    treatment["severity"]   = severity_label.capitalize() 
    treatment["pathogen"]   = entry.get("pathogen", "Unknown") 
    treatment["region"]     = REGION_NAMES.get(region, "India") 
    treatment["helpline"]   = KVK_HELPLINE 

    # Hindi translation for key action (basic) 
    hindi_map = { 
        "mild":   "हल्का संक्रमण — नियनमत निगरािी करें", 
        "severe": "गंभीर संक्रमण — तुरंत कृषि सलाहकार से संपर्क करें"
    } 
    treatment["hindi_summary"] = hindi_map.get(severity_label, "") 

    return treatment
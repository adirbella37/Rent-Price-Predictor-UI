from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from assets_data_prep import prepare_data

app = Flask(__name__)
model = joblib.load("trained_model.pkl")  # מודל עם pipeline מלא

@app.route('/')
def home():
    return render_template('index.html', form={}, errors={})

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    errors = {}

    required_fields = {
        'neighborhood': 'יש להזין שכונה',
        'area': 'יש להזין שטח',
        'propertyType': 'יש להזין סוג נכס',
        'address': 'יש להזין כתובת',
        'rooms': 'יש להזין מספר חדרים',
        'floor': 'יש להזין קומה',
        'totalFloors': 'יש להזין מספר קומות בבניין',
        'arnona': 'יש להזין ארנונה חודשית',
        'vaadBayit': 'יש להזין דמי ועד בית'
    }

    for field, msg in required_fields.items():
        if not form.get(field, '').strip():
            errors[field] = msg

    # בדיקת ערך שטח
    area_str = form.get('area', '').strip()
    try:
        area_val = float(area_str)
        if area_val < 10:
            errors['area'] = 'לא ניתן להזין שטח דירה קטן מ-10'
    except ValueError:
        errors['area'] = 'יש להזין מספר חוקי בשדה שטח'

    # אם יש שגיאות – הצגה עם הודעות שגיאה מותאמות
    if errors:
        return render_template('index.html',
                               errors=errors,
                               form=form,
                               general_error='אנא תקן את השדות המסומנים')

    # קלטי טופס – שטח גינה עם ברירת מחדל 0
    garden_area_val = float_or_zero(form.get('garden_area'))

    data = {
        'neighborhood': form.get('neighborhood'),
        'area': area_val,
        'property_type': form.get('propertyType'),
        'address': form.get('address'),
        'room_num': float(form.get('rooms')),
        'floor': float(form.get('floor')),
        'total_floors': float(form.get('totalFloors')),
        'monthly_arnona': float(form.get('arnona')),
        'building_tax': float(form.get('vaadBayit')),
        'garden_area': garden_area_val,
        'distance_from_center': np.nan,
        'description': '',
        'has_parking': 'חניה' in form,
        'has_storage': 'מחסן' in form,
        'elevator': 'מעלית' in form,
        'ac': 'מזגן' in form,
        'handicap': 'נגישה לנכים' in form,
        'has_bars': 'סורגים' in form,
        'has_safe_room': 'ממ"ד' in form,
        'has_balcony': 'מרפסת' in form,
        'is_furnished': 'מרוהטת' in form,
        'is_renovated': 'משופצת' in form
    }

    df = pd.DataFrame([data])

    try:
        processed = prepare_data(df, "test")
        if processed.empty:
            return render_template('index.html',
                                   form=form,
                                   errors={},
                                   general_error="יש בעיה בהזנת הנתונים")
        prediction = model.predict(processed)[0]
        return render_template('index.html',
                               prediction_text=f"שכר הדירה החזוי הוא: ₪{prediction:,.0f}",
                               form={},
                               errors={})
    except Exception as e:
        return render_template('index.html',
                               form=form,
                               errors={},
                               general_error=f"שגיאה במהלך החיזוי: {str(e)}")

def float_or_zero(val):
    try:
        return float(val)
    except:
        return 0.0

if __name__ == '__main__':
    app.run(debug=True)

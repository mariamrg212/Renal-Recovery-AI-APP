import streamlit as st
import pandas as pd
import numpy as np
import joblib # تم إلغاء التعليق لتحميل النموذج

st.set_page_config(page_title="Renal recovery in diabetic AKI", layout="centered")

# --- Data Loading and Cleanup ---
try:
    # 1. تحميل ملف البيانات الفعلي الذي تم تسميته Main data 2.csv
    data_df = pd.read_csv('Main data 2.csv')
    st.sidebar.success("تم تحميل البيانات الأصلية بنجاح.")

    # 2. تحديد الأعمدة المطلوبة للنموذج (16 عموداً)
    data_df = data_df[[
        'Duration.of.DM.y', 'Sepsis', 'CKD', 'IHD', 'Volume.depletion', 
        'Conservative.TTT', 'Incremental.HD', 'Hb.admission', 'PTH.admission', 
        'PH', 'Creat.disch', 'Stage.3', 'HbA1c', 'Sofa.score', 
        'SAPS.II.score', 'Ventilated'
    ]]

    # 3. إعادة تسمية الأعمدة حسب الطلب
    data_df = data_df.rename(columns={
        'Sofa.score':'Sofa score',
        'Duration.of.DM.y':'DM duration',
        'Stage.3':'Stage 3 AKI',
        'Creat.disch':'Serum creatinine at discharge',
        'Ventilated':'Mechanical ventilation',
        'Volume.depletion':'Prerenal Failure',
        'Conservative.TTT':'Conservative treatment',
        'SAPS.II.score':'SAPS II score',
        'Incremental.HD':'RRT (Renal Replacement Therapy)', # تم تحديثها لتعكس تغيير الـ UI
        'PTH.admission':'NGAL (Admission)', # تم تحديثها لتعكس تغيير الـ UI
        'PH':'Blood pH',
        'Hb.admission':'Hb level (Admission)' # تم تحديثها لتعكس تغيير الـ UI
    })
    
    # تحويل الأعمدة الرقمية لضمان عمل min/max/median بشكل صحيح
    for col in data_df.columns:
        if data_df[col].dtype == 'object':
             # محاولة تحويل الأعمدة التي يفترض أن تكون رقمية ولكن تم قراءتها كـ object
            try:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            except:
                pass # تجاهل إذا لم تكن قابلة للتحويل (مثل الأعمدة الثنائية)
    
    # ملء القيم المفقودة (إذا وجدت) بالوسيط لضمان عمل min/max/median
    data_df = data_df.fillna(data_df.median(numeric_only=True))

except FileNotFoundError:
    st.error("خطأ: لم يتم العثور على ملف البيانات 'Main data 2.csv'. يرجى التأكد من وجوده وتشغيل التطبيق مرة أخرى.")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء معالجة ملف البيانات: {e}")
    st.stop()

# --- Model Loading (Using actual joblib) ---
# تم تغيير اسم الملف إلى 'Random Forest Model.pkl'
MODEL_FILE = 'Random Forest Model.pkl'

try:
    # محاولة تحميل ملف النموذج الفعلي
    model = joblib.load(MODEL_FILE)
    st.sidebar.success(f"تم تحميل النموذج '{MODEL_FILE}' بنجاح.")
except FileNotFoundError:
    st.error(f"خطأ: لم يتم العثور على ملف النموذج '{MODEL_FILE}'. يرجى التأكد من أنه في نفس المجلد مع ملف app.py.")
    st.stop() 
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
    st.stop()
# --- End of Model Loading ---


st.title('Renal recovery in diabetic AKI')
st.markdown("---")
st.write("This application predicts the likelihood of **Renal Recovery** based on patient clinical and admission data.")

st.sidebar.header("Patient Input Data")
st.sidebar.markdown("Adjust the parameters below to get a prediction.")

# Dictionary to hold all input data
input_features = {}


# =========================================================================
# 1. DM Duration (Continuous) - Text Input
# =========================================================================
dm_duration_median = float(data_df['DM duration'].median())
dm_duration_str = st.sidebar.text_input(
    'DM duration (Years)',
    value=f"{dm_duration_median:.1f}", # Default to median, formatted to one decimal place
    key='dm_duration_input'
)
try:
    input_features['DM duration'] = float(dm_duration_str)
except ValueError:
    st.sidebar.error("DM duration must be a valid number.")
    input_features['DM duration'] = dm_duration_median 
    
# 2. Sepsis (Binary)
input_features['Sepsis'] = st.sidebar.checkbox("Sepsis Diagnosis", False)

# 3. CKD (Binary)
input_features['CKD'] = st.sidebar.checkbox("Chronic Kidney Disease (CKD)", False)

# 4. IHD (Binary)
input_features['IHD'] = st.sidebar.checkbox("Ischemic Heart Disease (IHD)", False)

# 5. Prerenal Failure (Binary)
input_features['Prerenal Failure'] = st.sidebar.checkbox("Prerenal Failure (Volume Depletion)", False)

# 6. Conservative treatment (Binary)
input_features['Conservative treatment'] = st.sidebar.checkbox("Conservative Treatment (No RRT)", False)

# 7. RRT (Binary)
input_features['RRT (Renal Replacement Therapy)'] = st.sidebar.checkbox("RRT (Renal Replacement Therapy)", False)


# =========================================================================
# 8. Hb level (Continuous) - Text Input
# =========================================================================
hb_median = float(data_df['Hb level (Admission)'].median())
hb_str = st.sidebar.text_input(
    'Hb level (Admission)',
    value=f"{hb_median:.1f}", 
    key='hb_level_input'
)
try:
    input_features['Hb level (Admission)'] = float(hb_str)
except ValueError:
    st.sidebar.error("Hb level (Admission) must be a valid number.")
    input_features['Hb level (Admission)'] = hb_median


# =========================================================================
# 9. NGAL (Continuous) - Text Input
# =========================================================================
ngal_median = float(data_df['NGAL (Admission)'].median())
ngal_str = st.sidebar.text_input(
    'NGAL (Admission)',
    value=f"{ngal_median:.0f}", # Display as integer (no decimals)
    key='ngal_input'
)
try:
    input_features['NGAL (Admission)'] = float(ngal_str)
except ValueError:
    st.sidebar.error("NGAL (Admission) must be a valid number.")
    input_features['NGAL (Admission)'] = ngal_median


# =========================================================================
# 10. Blood pH (Continuous) - Text Input
# =========================================================================
ph_median = float(data_df['Blood pH'].median())
ph_str = st.sidebar.text_input(
    'Blood pH (Admission)',
    value=f"{ph_median:.2f}", 
    key='ph_input'
)
try:
    input_features['Blood pH'] = float(ph_str)
except ValueError:
    st.sidebar.error("Blood pH (Admission) must be a valid number.")
    input_features['Blood pH'] = ph_median


# =========================================================================
# 11. Serum creatinine at discharge (Continuous) - Text Input
# =========================================================================
creat_median = float(data_df['Serum creatinine at discharge'].median())
creat_str = st.sidebar.text_input(
    'Serum creatinine at discharge (mg/dL)',
    value=f"{creat_median:.1f}",
    key='creat_input'
)
try:
    input_features['Serum creatinine at discharge'] = float(creat_str)
except ValueError:
    st.sidebar.error("Serum creatinine at discharge must be a valid number.")
    input_features['Serum creatinine at discharge'] = creat_median


# 12. Stage 3 AKI (Binary)
input_features['Stage 3 AKI'] = st.sidebar.checkbox("AKI Stage 3 at Admission", True)


# =========================================================================
# 13. HbA1c (Continuous) - Text Input
# =========================================================================
hba1c_median = float(data_df['HbA1c'].median())
hba1c_str = st.sidebar.text_input(
    'HbA1c',
    value=f"{hba1c_median:.1f}",
    key='hba1c_input'
)
try:
    input_features['HbA1c'] = float(hba1c_str)
except ValueError:
    st.sidebar.error("HbA1c must be a valid number.")
    input_features['HbA1c'] = hba1c_median


# =========================================================================
# 14. Sofa score (Continuous) - Text Input
# =========================================================================
sofa_median = int(data_df['Sofa score'].median())
sofa_str = st.sidebar.text_input(
    'SOFA score',
    value=f"{sofa_median}", # Display as integer
    key='sofa_input'
)
try:
    input_features['Sofa score'] = int(float(sofa_str)) # Convert to float first, then int for robustness
except ValueError:
    st.sidebar.error("SOFA score must be a valid integer number.")
    input_features['Sofa score'] = sofa_median


# =========================================================================
# 15. SAPS II score (Continuous) - Text Input
# =========================================================================
saps_median = int(data_df['SAPS II score'].median())
saps_str = st.sidebar.text_input(
    'SAPS II score',
    value=f"{saps_median}", # Display as integer
    key='saps_input'
)
try:
    input_features['SAPS II score'] = int(float(saps_str)) # Convert to float first, then int for robustness
except ValueError:
    st.sidebar.error("SAPS II score must be a valid integer number.")
    input_features['SAPS II score'] = saps_median


# 16. Mechanical ventilation (Binary)
input_features['Mechanical ventilation'] = st.sidebar.checkbox("Requires Mechanical Ventilation", False)


# --- Prediction Logic ---
# الترتيب: DM duration, Sepsis, CKD, IHD, Prerenal Failure, Conservative treatment, RRT, Hb level, NGAL, Blood pH, Creat. disch, Stage 3 AKI, HbA1c, Sofa score, SAPS II score, Ventilated 
# (يجب أن يتطابق هذا الترتيب تماماً مع ترتيب الميزات التي تم تدريب النموذج عليها)

input_array = np.array([[
    input_features['DM duration'],
    1 if input_features['Sepsis'] else 0,
    1 if input_features['CKD'] else 0,
    1 if input_features['IHD'] else 0,
    1 if input_features['Prerenal Failure'] else 0,
    1 if input_features['Conservative treatment'] else 0,
    1 if input_features['RRT (Renal Replacement Therapy)'] else 0,
    input_features['Hb level (Admission)'],
    input_features['NGAL (Admission)'],
    input_features['Blood pH'],
    input_features['Serum creatinine at discharge'],
    1 if input_features['Stage 3 AKI'] else 0,
    input_features['HbA1c'],
    input_features['Sofa score'],
    input_features['SAPS II score'],
    1 if input_features['Mechanical ventilation'] else 0 
]])

# Ensure the array is 2D for prediction
input_array = input_array.reshape(1, -1)

if st.sidebar.button('Predict Renal Recovery'):
    try:
        # Perform prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        # Interpret results
        result = "Yes (Full/Partial Recovery)" if prediction == 1 else "No (RRT Dependence)"
        recovery_proba = prediction_proba[1] * 100
        non_recovery_proba = prediction_proba[0] * 100

        st.markdown("## Prediction Result")
        
        if prediction == 1:
            st.success(f"**Prediction: {result}**")
            st.balloons()
        else:
            st.error(f"**Prediction: {result}**")
            
        st.markdown(f"The model predicts a **{recovery_proba:.1f}% chance** of Renal Recovery.")

        st.markdown("### Probability Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recovery Probability", f"{recovery_proba:.1f}%", f"{recovery_proba - non_recovery_proba:.1f}% difference")
        with col2:
            st.metric("Non-Recovery Probability", f"{non_recovery_proba:.1f}%", f"{non_recovery_proba - recovery_proba:.1f}% difference")
            
        st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("يرجى مراجعة سجل الأخطاء (console) للحصول على مزيد من التفاصيل.")

# Displaying inputs for verification
st.sidebar.markdown("---")
st.sidebar.caption("Input Array Order:")
st.sidebar.code(list(data_df.columns))

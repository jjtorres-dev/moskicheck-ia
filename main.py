from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# 1. Inicialización de la API
app = FastAPI(
    title="MoskiCheck IA API",
    description="API de Microservicio para la predicción de Dengue, Zika y Chikungunya mediante Machine Learning",
    version="1.0.0"
)

# 2. Carga Global de los Modelos Entrenados
try:
    modelo = joblib.load('modelo_diagnostico.pkl')
    encoder = joblib.load('encoder.pkl')
    print("✅ Archivos 'modelo_diagnostico.pkl' y 'encoder.pkl' cargados con éxito.")
except Exception as e:
    print(f"❌ Error crítico al cargar los archivos del modelo: {e}")
    modelo = None
    encoder = None

# 3. Definición del Modelo de Datos (Esquema del JSON de entrada)
class Consulta_Sintomas(BaseModel):
    fever: int = Field(..., description="Fiebre: 1 si presenta, 0 si no", ge=0, le=1)
    headache: int = Field(..., description="Dolor de cabeza: 1 si presenta, 0 si no", ge=0, le=1)
    joint_pain: int = Field(..., description="Dolor articular: 1 si presenta, 0 si no", ge=0, le=1)
    muscle_pain: int = Field(..., description="Dolor muscular: 1 si presenta, 0 si no", ge=0, le=1)
    vomiting: int = Field(..., description="Vómitos: 1 si presenta, 0 si no", ge=0, le=1)
    rash: int = Field(..., description="Sarpullido: 1 si presenta, 0 si no", ge=0, le=1)
    fatigue: int = Field(..., description="Fatiga: 1 si presenta, 0 si no", ge=0, le=1)
    eye_pain: int = Field(..., description="Dolor de ojos: 1 si presenta, 0 si no", ge=0, le=1)
    nausea: int = Field(..., description="Náuseas: 1 si presenta, 0 si no", ge=0, le=1)
    chills: int = Field(..., description="Escalofríos: 1 si presenta, 0 si no", ge=0, le=1)
    bleeding: int = Field(..., description="Sangrado: 1 si presenta, 0 si no", ge=0, le=1)
    red_eyes: int = Field(..., description="Ojos rojos: 1 si presenta, 0 si no", ge=0, le=1)
    joint_swelling: int = Field(..., description="Articulaciones hinchadas: 1 si presenta, 0 si no", ge=0, le=1)
    itching: int = Field(..., description="Picazón: 1 si presenta, 0 si no", ge=0, le=1)

# 4. Ruta de Diagnóstico (Endpoint)
@app.post("/predecir", summary="Generar predicción diagnóstica basada en síntomas")
def predecir_enfermedad(sintomas: Consulta_Sintomas):
    # Verificación de la disponibilidad del modelo antes de procesar
    if modelo is None or encoder is None:
        raise HTTPException(
            status_code=503, 
            detail="El servicio de IA no está disponible temporalmente porque los modelos no se cargaron."
        )
    
    try:
        # Transformar los datos de entrada a un diccionario compatible con Pydantic v2
        datos_dict = sintomas.model_dump()
        
        # Crear la lista ordenada de características para evitar que las columnas se desalineen
        caracteristicas_ordenadas = [
            "fever", "headache", "joint_pain", "muscle_pain", "vomiting", 
            "rash", "fatigue", "eye_pain", "nausea", "chills", 
            "bleeding", "red_eyes", "joint_swelling", "itching"
        ]
        
        # Estructurar el DataFrame con las columnas explícitamente ordenadas
        datos_df = pd.DataFrame([datos_dict], columns=caracteristicas_ordenadas)
        
        # Ejecutar la predicción del modelo
        prediccion_numerica = modelo.predict(datos_df)
        
        # Decodificar el número devuelto para obtener el nombre de la enfermedad
        enfermedad_resultado = encoder.inverse_transform(prediccion_numerica)[0]
        
        # Calcular las probabilidades individuales para cada clase
        probabilidades_raw = modelo.predict_proba(datos_df)[0]
        etiquetas_clases = encoder.inverse_transform(modelo.classes_)
        
        # Construir el mapeo de porcentajes redondeados a 4 decimales
        mapa_probabilidades = {
            str(etiquetas_clases[i]): round(float(probabilidades_raw[i]), 4)
            for i in range(len(etiquetas_clases))
        }
        
        # Extraer el nivel de confianza asociado específicamente a la enfermedad ganadora
        indice_enfermedad = list(etiquetas_clases).index(enfermedad_resultado)
        confianza_resultado = round(float(probabilidades_raw[indice_enfermedad]), 4)
        
        # Respuesta estructurada final
        return {
            "status": "success",
            "prediction": enfermedad_resultado,
            "confidence": confianza_resultado,
            "probabilities": mapa_probabilidades
        }
        
    except Exception as error:
        raise HTTPException(
            status_code=500, 
            detail=f"Ocurrió un error interno durante el procesamiento de la predicción: {str(error)}"
        )

# 5. Ruta de Verificación de Estado (Health Check)
@app.get("/health", summary="Verificar el estado del servidor")
def health_check():
    return {
        "status": "online",
        "models_loaded": modelo is not None and encoder is not None
    }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import time
from datetime import date

# ======================
# GLOBALS
# ======================

MODEL = None
FEATURES = []
NAME2CODE = {}
CODE2NAME = {}
HIST_DF = None
LOAD_ERR = ""
START_TS = time.time()

app = FastAPI(title="Predicción de Demanda Diario", version="2.0")


# ======================
# 1. MODELO Pydantic
# ======================

class RangeInput(BaseModel):
    tipo_articulo_name: Optional[str] = None
    tipo_articulo_cod: Optional[int] = None
    fecha_inicio: date
    fecha_fin: date

    @model_validator(mode="after")
    def validar(self):
        if self.tipo_articulo_name is None and self.tipo_articulo_cod is None:
            raise ValueError("Debes enviar tipo_articulo_name o tipo_articulo_cod.")
        if self.fecha_fin < self.fecha_inicio:
            raise ValueError("fecha_fin no puede ser menor que fecha_inicio.")
        return self


class RangeDayPrediction(BaseModel):
    fecha: date
    tipo_articulo_codigo: int
    tipo_articulo_nombre: str
    demanda_pronosticada: float


class RangeResponse(BaseModel):
    predictions: List[RangeDayPrediction]


# ======================
# 2. LOAD ARTIFACTS
# ======================

@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURES, NAME2CODE, CODE2NAME, HIST_DF, LOAD_ERR

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "Model")

        MODEL = joblib.load(os.path.join(model_dir, "xgboost_demanda_model.pkl"))
        FEATURES = joblib.load(os.path.join(model_dir, "features_list.pkl"))
        NAME2CODE = joblib.load(os.path.join(model_dir, "sku_mapeo_nombre_a_codigo.pkl"))
        CODE2NAME = {v: k for k, v in NAME2CODE.items()}

        # Cargar histórico
        HIST_DF = joblib.load(os.path.join(model_dir, "historico.pkl"))
        HIST_DF["fecha_pedido"] = pd.to_datetime(HIST_DF["fecha_pedido"])
        HIST_DF = HIST_DF.sort_values(["tipo_articulo_cod", "fecha_pedido"]).reset_index(drop=True)

        print("✅ Artefactos cargados correctamente.")

    except Exception as e:
        LOAD_ERR = f"{type(e).__name__}: {e}"
        print("🚨 Error cargando artefactos:", LOAD_ERR)


# ======================
# 3. HELPERS
# ======================

def resolve_sku_code(name: Optional[str], code: Optional[int]) -> int:
    if code is not None:
        return int(code)
    if name not in NAME2CODE:
        raise HTTPException(status_code=400, detail=f"SKU '{name}' no existe en el modelo.")
    return int(NAME2CODE[name])


def forecast_range(sku_code: int, fecha_ini: date, fecha_fin: date):
    """
    Predicción día por día usando:
    - histórico real
    - predicciones para días futuros
    """
    if HIST_DF is None:
        raise HTTPException(status_code=503, detail="Histórico no disponible.")

    hist = HIST_DF[HIST_DF["tipo_articulo_cod"] == sku_code].copy()

    if hist.empty:
        raise HTTPException(status_code=400, detail=f"No hay histórico para el SKU {sku_code}.")

    hist = hist.sort_values("fecha_pedido").reset_index(drop=True)

    # Filtramos solo lo anterior a fecha_ini (no futuro)
    hist = hist[hist["fecha_pedido"] < pd.to_datetime(fecha_ini)]

    if hist.empty:
        raise HTTPException(status_code=400, detail="No hay datos previos a fecha_inicio para calcular lags.")

    preds = []

    fechas = pd.date_range(fecha_ini, fecha_fin, freq="D")

    for f in fechas:
        # Calcular lags
        lag_1 = hist["cantidad"].iloc[-1]
        lag_2 = hist["cantidad"].iloc[-2] if len(hist) >= 2 else lag_1
        rolling_3 = hist["cantidad"].tail(3).mean() if len(hist) >= 3 else lag_1

        # Feature row
        row = {
            "tipo_articulo_cod": sku_code,
            "anio": f.year,
            "mes": f.month,
            "dia": f.day,
            "dia_semana": f.weekday(),
            "semana_mes": (f.day - 1) // 7 + 1,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "rolling_3": rolling_3,
        }

        df_row = pd.DataFrame([row])[FEATURES]
        y_pred = MODEL.predict(df_row.values)[0]

        preds.append(
            RangeDayPrediction(
                fecha=f.date(),
                tipo_articulo_codigo=sku_code,
                tipo_articulo_nombre=CODE2NAME[sku_code],
                demanda_pronosticada=round(float(y_pred), 2),
            )
        )

        # Expandir historial con la predicción
        hist = pd.concat([
            hist,
            pd.DataFrame({
                "fecha_pedido": [f],
                "tipo_articulo_cod": [sku_code],
                "cantidad": [y_pred]
            })
        ], ignore_index=True)

    return preds


# ======================
# 4. ENDPOINT
# ======================

@app.post("/predict_range", response_model=RangeResponse)
def predict_range(body: RangeInput):

    sku_code = resolve_sku_code(body.tipo_articulo_name, body.tipo_articulo_cod)

    result = forecast_range(
        sku_code=sku_code,
        fecha_ini=body.fecha_inicio,
        fecha_fin=body.fecha_fin
    )

    return RangeResponse(predictions=result)

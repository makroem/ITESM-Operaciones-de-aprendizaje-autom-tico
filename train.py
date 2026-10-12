"""Entrenamiento reproducible de regresión para Wine Quality."""
import os

# Se fijan antes de importar las bibliotecas numéricas.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]


def argumentos():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "elasticnet"], default="linear",
                        help="linear conserva el modelo original; elasticnet añade regularización")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Intensidad de regularización de ElasticNet, mayor que cero (0.01)")
    parser.add_argument("--l1-ratio", type=float, default=0.5,
                        help="Proporción L1 de ElasticNet entre 0 y 1 (0.5)")
    parser.add_argument("--fit-intercept", type=int, choices=[0, 1], default=1,
                        help="Estimar intercepto: 1 sí, 0 no (1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de las particiones y modelo (42)")
    parser.add_argument("--data", default="winequality-red.csv", help="CSV separado por punto y coma")
    parser.add_argument("--experiment", default="C5061_vino_regresion_parametrizada",
                        help="Nombre del experimento de MLflow")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="Base de MLflow")
    parser.add_argument("--run-name", default="baseline_lineal", help="Nombre visible de la corrida")
    parser.add_argument("--stage", choices=["comparacion", "reproducibilidad", "final"],
                        default="comparacion", help="Propósito de la corrida")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluar prueba solo para el ganador")
    parser.add_argument("--result-file", default="ultima_corrida.json", help="Resumen de salida")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("alpha debe ser finito y mayor que cero")
    if not 0 <= args.l1_ratio <= 1:
        parser.error("l1-ratio debe estar entre 0 y 1")
    if not 0 <= args.seed < 2**32:
        parser.error("seed debe estar entre 0 y 2**32 - 1")
    if args.evaluate_test != (args.stage == "final"):
        parser.error("La etapa final requiere --evaluate-test; las otras etapas no lo permiten")
    return args


def cargar_datos(ruta):
    archivo = Path(ruta)
    if not archivo.exists():
        archivo.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL, archivo)
    original = pd.read_csv(archivo, sep=";")
    if set(original.columns) != set(FEATURES + ["quality"]):
        raise ValueError("El CSV no tiene las 11 variables y quality esperadas")
    datos = original.apply(pd.to_numeric, errors="coerce")
    datos = datos.dropna(subset=["quality"]).drop_duplicates().reset_index(drop=True)
    datos["quality"] = datos["quality"].astype(int)
    return datos, {"sha256_csv": hashlib.sha256(archivo.read_bytes()).hexdigest(),
                   "filas_originales": len(original), "filas_limpias": len(datos),
                   "filas_eliminadas": len(original) - len(datos)}


def dividir_datos(datos, seed):
    X, y = datos[FEATURES], datos["quality"]
    # Se conserva el 20 % de prueba y seed=42 del notebook original.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    # 25 % del 80 % = 20 % del total para validación; queda 60 % para entrenar.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def crear_modelo(args):
    if args.model == "linear":
        regresor = LinearRegression(fit_intercept=bool(args.fit_intercept))
    else:
        regresor = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio,
                             fit_intercept=bool(args.fit_intercept), random_state=args.seed,
                             selection="cyclic", max_iter=10000, tol=1e-7)
    # Mediana y escalado se aprenden exclusivamente con entrenamiento.
    return make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), regresor)


def evaluar(y, pred, prefijo):
    return {f"mae_{prefijo}": float(mean_absolute_error(y, pred)),
            f"rmse_{prefijo}": float(np.sqrt(mean_squared_error(y, pred))),
            f"r2_{prefijo}": float(r2_score(y, pred))}


def guardar_artefactos(carpeta, metricas, y_val, pred, particiones):
    (carpeta / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    pd.DataFrame({"indice": y_val.index, "real": y_val.to_numpy(),
                  "prediccion": pred}).to_csv(carpeta / "predicciones_validacion.csv", index=False)
    (carpeta / "particiones.json").write_text(json.dumps(particiones, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_val, pred, alpha=0.4)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
    ax.set(xlabel="Calidad real", ylabel="Calidad estimada", title="Predicciones en validación")
    fig.tight_layout()
    fig.savefig(carpeta / "predicciones.png", dpi=150)
    plt.close(fig)


def main():
    args = argumentos()
    random.seed(args.seed)
    np.random.seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    experimento = mlflow.set_experiment(args.experiment)
    datos, informacion = cargar_datos(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(datos, args.seed)
    particiones = {"train": X_train.index.tolist(), "validation": X_val.index.tolist(),
                   "test": X_test.index.tolist()}
    modelo = crear_modelo(args)
    with mlflow.start_run(run_name=args.run_name) as run:
        parametros = {"model": args.model, "fit_intercept": args.fit_intercept, "seed": args.seed,
                      "test_size": 0.20, "validation_fraction_dev": 0.25,
                      "imputacion": "median", "escalado": "RobustScaler",
                      "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                      **informacion}
        if args.model == "elasticnet":
            parametros.update(alpha=args.alpha, l1_ratio=args.l1_ratio, max_iter=10000,
                              tol=1e-7, selection="cyclic")
        for clave, valor in parametros.items():
            mlflow.log_param(clave, valor)
        mlflow.set_tags({"etapa": args.stage, "sha256_script": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(), "python_version": sys.version.split()[0]})
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_val)
        metricas = evaluar(y_val, pred, "val")
        metricas.update(evaluar(y_train, modelo.predict(X_train), "train"))
        ingenuo = DummyRegressor(strategy="mean").fit(X_train, y_train)
        metricas["mae_media_val"] = float(mean_absolute_error(y_val, ingenuo.predict(X_val)))
        if args.evaluate_test:
            metricas.update(evaluar(y_test, modelo.predict(X_test), "test"))
        for clave, valor in metricas.items():
            mlflow.log_metric(clave, valor)
        with tempfile.TemporaryDirectory() as temporal:
            carpeta = Path(temporal)
            guardar_artefactos(carpeta, metricas, y_val, pred, particiones)
            for archivo in carpeta.iterdir():
                mlflow.log_artifact(str(archivo), artifact_path="evaluacion")
        for archivo in [Path(__file__), Path(args.data), Path("requirements.txt"),
                        Path("requirements-lock.txt"), Path("entorno.json")]:
            if archivo.exists():
                mlflow.log_artifact(str(archivo), artifact_path="reproducibilidad")
        info_modelo = mlflow.sklearn.log_model(
            modelo, name="modelo", input_example=X_train.head(3),
            serialization_format="cloudpickle", pip_requirements="requirements.txt")
        resumen = {"run_id": run.info.run_id, "experiment_id": experimento.experiment_id,
                   "run_name": args.run_name, "model_uri": info_modelo.model_uri,
                   "parametros": parametros, "metricas": metricas}
    destino = Path(args.result_file)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    main()

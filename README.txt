C5061 - Experimentos reproducibles con MLflow
Marco Emilio Jimenez Jimenez - A01797948
Notebook: C5061_Experimentos_MLflow_JimenezMarco.ipynb

OBJETIVO Y ARCHIVOS
Predecir la calidad del vino tinto con regresión lineal y comparar variantes
ElasticNet. train.py contiene la carga, limpieza, entrenamiento, evaluación y
registro en MLflow. requirements.txt fija las dependencias; este README
describe la ejecución de la versión adjunta del notebook.

EJECUCIÓN EN GOOGLE COLAB
1. Abre el notebook y selecciona Runtime > Change runtime type > Python 3
   con CPU. Ejecuta la primera celda para crear /content/c5061_mlflow.
2. Esta versión necesita requirements.txt antes de instalar las bibliotecas.
   Agrega una celda después de la primera y sube requirements.txt y este
   README.txt a la carpeta de trabajo ejecutando:

   from google.colab import files
   files.upload()

3. Ejecuta "Instalar versiones fijas". Después de instalar por primera vez,
   usa Runtime > Restart session y vuelve a ejecutar desde el inicio para
   cargar las versiones nuevas. Los archivos permanecen en la sesión.
4. Ejecuta las celdas en orden. "Crear el script de entrenamiento" escribe
   train.py. Luego se prueba la ayuda, se ejecuta el baseline sin argumentos
   y se lanzan seis configuraciones adicionales. El CSV se descarga de UCI.
5. Continúa con la tabla comparativa, las dos repeticiones para comprobar
   reproducibilidad, la evaluación del ganador en prueba y el registro del
   modelo. El lote completo genera diez corridas, siete configuraciones distintas.
6. Ejecuta "MLflow en Google Colab". La interfaz aparece dentro del notebook
   en el puerto 5001. Si está ocupado, cambia PUERTO a 5002.
7. La celda incluye un host interno específico de una sesión anterior.
   Si aparece Invalid Host header, consulta mlflow_server.log y sustituye
   el valor de la línea hosts += por el Host rechazado exacto, con su puerto.
   Reejecuta toda esa celda; no repitas los entrenamientos.
8. En MLflow abre C5061_vino_regresion_parametrizada y filtra:

   tags.etapa = 'comparacion'

   Muestra alpha, l1_ratio, fit_intercept, seed, mae_val, rmse_val y r2_val.
   Ordena mae_val de menor a mayor; selecciona las siete corridas y pulsa Compare.
   Guarda capturas de la tabla, Compare y Artifacts de una corrida.
9. Revisa el análisis generado en "Analizar los resultados". Antes de las
   últimas dos celdas, sube tus capturas con esta celda adicional:

   from google.colab import files
   import os
   os.makedirs('evidencias', exist_ok=True)
   os.chdir('evidencias')
   files.upload()
   os.chdir(PROYECTO)

10. Ejecuta las últimas dos celdas para crear README.md y descargar el ZIP.
    Conserva también este README.txt en el respaldo. Descarga el notebook
    ejecutado con File > Download > Download .ipynb antes de cerrar Colab.

COMANDOS DEL SCRIPT
Desde la carpeta que contiene train.py y requirements.txt:

   python -m pip install -r requirements.txt
   python train.py --help
   python train.py
   python train.py --model elasticnet --alpha 0.01 --l1-ratio 0.1 --fit-intercept 1 --seed 42 --experiment C5061_vino_regresion_parametrizada --run-name E04_l1_01 --result-file resumenes/E04_l1_01.json

En Colab, ejecutar desde /content/c5061_mlflow y añadir ! delante de los
comandos python. Para ejecución local se recomienda un entorno virtual;
la comprobación local inicial se realizó con Python 3.12.14.
Para abrir MLflow localmente, desde esa misma carpeta:

   python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Visita http://127.0.0.1:5000. En Colab utiliza la interfaz dentro del notebook.

PARÁMETROS Y REPRODUCIBILIDAD
Valores por defecto: model=linear, alpha=0.01, l1-ratio=0.5, fit-intercept=1
y seed=42. Alpha y l1-ratio se aplican únicamente a ElasticNet.
Se usa aproximadamente 60 % para entrenamiento, 20 % para validación y
20 % para prueba. Selecciona por menor MAE de validación. Para evaluar al
ganador en prueba, añade --stage final --evaluate-test a su comando.
La semilla controla random, NumPy, particiones y ElasticNet. Conserva datos,
código y versiones. Se comprueban métricas y predicciones con atol=1e-10,
rtol=0; identificadores y tiempos de corrida pueden cambiar.

SALIDAS Y ENTREGA
mlflow.db y mlruns/: corridas, artefactos y modelos.
comparacion.csv y analisis_resultados.md: evidencia de selección.
entorno.json y requirements-lock.txt: entorno y versiones utilizadas.
modelo_seleccionado.json: versión registrada, alias y corrida final.
resumenes/ y reproducibilidad.txt: resultados y comprobación de repetibilidad.
evidencias/: capturas reales de MLflow.
Entrega el notebook ejecutado, el ZIP con los archivos y el notebook de
exploración. Respalda SQLite y mlruns/ juntos. Las rutas de artefactos son
del entorno original; al cambiar de máquina, reconstruye las corridas con
el CSV conservado y los mismos argumentos.

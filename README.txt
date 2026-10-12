C5061 - Regresión de calidad del vino con MLflow

ARCHIVOS
train.py: entrenamiento parametrizado, evaluación y registro en MLflow.
requirements.txt: dependencias con versiones fijas.

GOOGLE COLAB
1. Abre C5061_Experimentos_MLflow_Colab_Corregido.ipynb.
2. Selecciona Runtime > Change runtime type > Python 3 > CPU > Save.
3. Ejecuta las secciones 1 y 2 para preparar /content/c5061_mlflow e instalar
   las dependencias. Reinicia una vez con Runtime > Restart session después
   de la instalación para cargar las versiones nuevas. No selecciones
   Disconnect and delete runtime. Después vuelve a ejecutar desde la sección 1.
4. Ejecuta las secciones 3 a 8 en orden. El notebook escribe train.py y
   requirements.txt, ejecuta siete configuraciones, comprueba dos repeticiones
   y evalúa al ganador. Los datos se descargan automáticamente desde UCI.
5. Usa la sección 9 corregida: inicia MLflow en el puerto 5001 y muestra
   la interfaz dentro del notebook. Autoriza el dominio del proxy de Colab
   y recupera del registro los nombres internos rechazados anteriormente.
6. Si aparece Invalid Host header, consulta mlflow_server.log. Agrega el valor
   exacto que sigue a "invalid Host header:" (incluido su puerto) con:

   hosts += ",NOMBRE_INTERNO:PUERTO"

   Coloca esa línea después de definir hosts y antes de definir origen;
   sustituye NOMBRE_INTERNO:PUERTO por el valor del registro. Vuelve a ejecutar
   toda la celda 9 para reiniciar el servidor con la autorización. El nombre
   puede cambiar entre sesiones; no uses el de una sesión anterior.
7. En MLflow abre el experimento C5061_vino_regresion_parametrizada y filtra
   tags.etapa = 'comparacion'. Selecciona las siete corridas y pulsa Compare.
   Toma capturas legibles de la tabla y Compare y súbelas en la sección 10.
8. Ejecuta las secciones 11 y 12: revisa el análisis y descarga el ZIP.
   Descarga también el notebook ejecutado con File > Download > Download .ipynb.

Si ya ejecutaste las corridas, corregir la interfaz no requiere entrenarlas
otra vez. Si falla la sección 6 tras instalar NumPy, reinicia la sesión y
recupera resultados desde corridas_comparacion.json antes de continuar.

EJECUCIÓN DEL SCRIPT POR SEPARADO
Guarda train.py y requirements.txt en la misma carpeta. En una terminal:

   python -m venv .venv

Activa el entorno: Windows PowerShell, .\.venv\Scripts\Activate.ps1;
Linux/macOS, source .venv/bin/activate. Después ejecuta:

   python -m pip install -r requirements.txt
   python train.py --help
   python train.py
   python train.py --model elasticnet --alpha 0.01 --l1-ratio 0.1 --fit-intercept 1 --seed 42 --run-name E04

En una celda de Colab, añade ! delante de los comandos python y ejecútalos
desde /content/c5061_mlflow. Instala antes de importar las bibliotecas.

Para consultar MLflow localmente desde la carpeta de entrenamiento:

   python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Abre http://127.0.0.1:5000. En Colab usa la interfaz de la sección 9 corregida.

PARÁMETROS Y REPRODUCIBILIDAD
Valores por defecto: model=linear, alpha=0.01, l1-ratio=0.5, fit-intercept=1,
seed=42. Alpha y l1-ratio se aplican a ElasticNet. Se usa aproximadamente
60 % para entrenamiento, 20 % para validación y 20 % para prueba.
Selecciona por menor MAE de validación. Para evaluar al ganador en prueba,
añade --stage final --evaluate-test a su comando; no cambies el ganador por prueba.
Conserva CSV, código, versiones y semilla. La comprobación compara métricas
y predicciones con tolerancia absoluta de 1e-10; los run_id pueden cambiar.

SALIDAS Y ENTREGA
mlflow.db y mlruns/ conservan corridas, artefactos y modelos; ultima_corrida.json
guarda el resumen más reciente. --result-file permite guardar otro resumen.
Entrega el notebook ejecutado, el ZIP generado y el notebook de exploración.
El ZIP debe incluir script, dependencias, README, análisis y capturas reales.
Respalda los archivos antes de cerrar Colab. El entrenamiento fue verificado
localmente con Python 3.12.14; el acceso al proxy se comprueba en tu sesión de Colab.

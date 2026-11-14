# Ajuste de Parametros con Diseño de Experimentos

Evaluación estadística del parámetro de temperatura inicial ($T_0$) del algoritmo de Recocido Simulado con vecindario 2-opt aplicado al Problema del Agente Viajero (TSP), utilizando un Diseño Factorial Completo $3 \times 3$.

## Diseño Experimental

* **Enfoque:** Diseño Factorial Completo de dos factores (9 puntos de diseño, 10 réplicas por punto, 90 ejecuciones totales).
* **Factor Manipulado ($T_0$):** Tres niveles derivados de una semilla fija (Bajo ($908.37$), Medio ($2378.37$) y Alto ($3848.37$).
* **Factor de Bloqueo ($n$):** Tres tamaños de instancia con $n \in \{5, 10, 15\}$ ciudades.
* **Análisis Estadístico:** ANOVA Tipo II ($\alpha = 0.05$) y estimación de medianas con intervalos de confianza del 95% mediante Bootstrap no paramétrico (2,000 remuestreos).

## Estructura y Ejecución

```text
.
├── main.py                  # Simulación TSP, ANOVA, Bootstrap y generación de gráficas
├── figure.png               # Gráficas de efectos principales e interacción
└── README.md                # Documentación del experimento
```

### Requisitos e Instalación

```bash
pip install numpy pandas matplotlib statsmodels
python main.py
```

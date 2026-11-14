import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

MATRICULA = 2148314
random.seed(MATRICULA)
np.random.seed(MATRICULA)
r = random.random()

T_MIN = 100.0
T_MAX = 5000.0
TAMANOS_N = [5, 10, 15]

def generate_tsp_instance(n: int, seed: int = 42):
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]

def tour_length(points, route) -> float:
    return sum(
        math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )

def two_opt_move(route, i: int, k: int):
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def simulated_annealing(points, initial_temp: float, seed: int, max_iter: int = 400) -> float:
    rng = random.Random(seed)
    n = len(points)
    
    current_route = list(range(n))
    rng.shuffle(current_route)
    current_length = tour_length(points, current_route)
    best_length = current_length
    
    temp = initial_temp
    final_temp = 1e-3
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iter)
    
    for _ in range(max_iter):
        i, k = sorted(rng.sample(range(n), 2))
        if k - i == 1 and i == 0 and k == n - 1:
            continue
        neighbor = two_opt_move(current_route, i, k)
        neighbor_length = tour_length(points, neighbor)
        
        delta = neighbor_length - current_length
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            current_route = neighbor
            current_length = neighbor_length
            if current_length < best_length:
                best_length = current_length
        temp *= cooling_rate
        
    return best_length

def bootstrap_median_ci(data, resamples: int = 2000, ci: float = 95.0):
    medians = [
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(resamples)
    ]
    lower = np.percentile(medians, (100.0 - ci) / 2.0)
    upper = np.percentile(medians, 100.0 - (100.0 - ci) / 2.0)
    return float(np.median(data)), float(lower), float(upper)

fracciones_A = {
    "bajo": 0.3 * r,
    "medio": 0.3 * r + 0.3,
    "alto": 0.3 * r + 0.6
}
niveles_T_A = {k: round(T_MIN + f * (T_MAX - T_MIN), 2) for k, f in fracciones_A.items()}

resultados_A = []
for n in TAMANOS_N:
    instancia = generate_tsp_instance(n, seed=42)
    for nivel, T0 in niveles_T_A.items():
        for rep in range(10):
            costo = simulated_annealing(instancia, T0, seed=MATRICULA + n * 100 + rep)
            resultados_A.append({
                "tamano": n,
                "nivel": nivel,
                "temperatura": T0,
                "replica": rep + 1,
                "costo": costo
            })

df_A = pd.DataFrame(resultados_A)

modelo_A = ols('costo ~ C(temperatura) * C(tamano)', data=df_A).fit()
tabla_anova_A = sm.stats.anova_lm(modelo_A, typ=2)
print("Tabla ANOVA de dos factores: \n")
print(tabla_anova_A)

print("\nTemperatura óptima y Bootstrap:\n")
resumen_A = []
for n in TAMANOS_N:
    df_n = df_A[df_A['tamano'] == n]
    stats_nivel = df_n.groupby(['nivel', 'temperatura'])['costo'].median().reset_index()
    stats_nivel = stats_nivel.sort_values(by=['costo', 'temperatura'], ascending=[True, True])
    mejor = stats_nivel.iloc[0]
    
    t_opt = mejor['temperatura']
    nivel_opt = mejor['nivel']
    muestra_opt = df_n[df_n['temperatura'] == t_opt]['costo'].values
    med, ci_lo, ci_hi = bootstrap_median_ci(muestra_opt)
    
    resumen_A.append({
        "n": n,
        "Nivel": nivel_opt,
        "T_opt": t_opt,
        "Mediana_Bootstrap": round(med, 2),
        "IC_95_Inferior": round(ci_lo, 2),
        "IC_95_Superior": round(ci_hi, 2)
    })

print(pd.DataFrame(resumen_A).to_string(index=False))

fig_A, axes_A = plt.subplots(1, 3, figsize=(16, 5))
temps_u = sorted(df_A['temperatura'].unique())
tams_u = sorted(df_A['tamano'].unique())

axes_A[0].boxplot(
    [df_A[df_A['temperatura'] == t]['costo'].values for t in temps_u],
    tick_labels=temps_u,
    patch_artist=True,
    boxprops=dict(facecolor='#BDD7EE')
)
axes_A[0].set_title("Efecto Principal: Temperatura (T₀)")
axes_A[0].set_xlabel("Temperatura Inicial (T₀)")
axes_A[0].set_ylabel("Costo de Ruta (Distancia)")

axes_A[1].boxplot(
    [df_A[df_A['tamano'] == n]['costo'].values for n in tams_u],
    tick_labels=tams_u,
    patch_artist=True,
    boxprops=dict(facecolor='#C6E0B4')
)
axes_A[1].set_title("Efecto Principal: Tamaño TSP (n)")
axes_A[1].set_xlabel("Número de Ciudades (n)")

interaction_plot(
    df_A['temperatura'],
    df_A['tamano'],
    df_A['costo'],
    ax=axes_A[2],
    markers=['D', '^', 'o']
)
axes_A[2].set_title("Interacción: Temperatura × Tamaño")
axes_A[2].set_xlabel("Temperatura Inicial (T₀)")
axes_A[2].set_ylabel("Costo Promedio")

plt.tight_layout()
plt.savefig("figure.png", dpi=300)
plt.close()

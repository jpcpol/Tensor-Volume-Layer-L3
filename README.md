# CAL-L3 — Tensor Volume Layer

**Parte de:** [CAL — Cognitive Abstraction Layers](../)  
**Autor:** Juan Pablo Chancay (Aural Syncro)  
**Estado:** En desarrollo — operador de composición C (problema abierto)  
**Venue objetivo:** NeurIPS / ICML

## Definición formal

```
L3: V = C({T⁽ˢ⁾}_{s=1}^{n})
```

Donde `C` es el **operador de composición** que mapea una colección de tensores
cognitivos (sesiones L2) en un volumen tensorial unificado `V`, preservando
relaciones causales, coherencia temporal y estabilidad dimensional.

## Requisitos del operador C (§5.2 CAL pre-paper)

1. **Causal preservation** — relaciones causales entre dimensiones codificables en V
2. **Temporal coherence** — índices temporales de T⁽ˢ⁾ composables en orden global
3. **Dimensional stability** — estructura 11-dim preservada; L4 aplica las mismas ops
4. **Tractability** — tamaño de V no crece linealmente con n

## Candidato inicial: Tucker decomposition

Estrategia de §5.3.2 del pre-paper. κ(V) = rango efectivo.
Librería: `tensorly`. Validación vía benchmark sintético con ground-truth causal.
Pre-registro de hipótesis requerido antes de correr experimentos.

## Estructura (en construcción)

```
L3/
├── README.md
├── paper/              ← paper L3 (en desarrollo)
├── src/                ← implementación operador C
│   └── composition/    ← Tucker/manifold candidates
├── benchmarks/         ← corpus sintético + ground-truth causal
└── experiments/        ← tests de conservación causal + manifold
```

## Dependencias

- Requiere corpus L2 validado (`../L2/`) como entrada del operador C
- Bloqueante de L4 Rol 2 (`../L4/`) — ver `cal-collaboration.md`
- Colaboración AMD-Instinct: S5 (kernel proxy M(V)) espera a que C exista

# Conecta / AIML339 — Resumen de progreso para check-in

## Pregunta de investigación
¿Una fusión aprendida de similitud por sentence-embeddings + features de compatibilidad estructuradas supera a un baseline de embeddings puros?

---

## 1. Qué está construido y validado

**Pipeline de datos (Track 1 — Speed Dating Experiment)**
- Parser propio para un CSV mal formado (encoding cp1252, dos formatos de fila mezclados) — recupera 8349/8378 filas (99.65%); las 29 restantes están corruptas en la fuente y quedan documentadas, no adivinadas.
- Deduplicación de pares direccionales: cada cita real generaba dos filas (una por participante); se colapsó a una fila por par, evitando pseudo-replicación y fuga de datos.
- Split de train/test agrupado por `wave` (no por participante) — verificado que ningún participante aparece en más de una wave, así que no hay fuga entre train y test.

**Features**
- Track 1: `same_field`, `same_career` (exact-match sobre códigos categóricos), `age_proximity` (decaimiento exponencial).
- Track 2: `same_cohort`, `arrival_date_proximity`, `course_overlap_jaccard`.
- `int_corr` (correlación de intereses pre-calculada) reservada solo para validación externa, nunca como input del modelo — evita circularidad.
- `samerace` y nacionalidad excluidas de todo input, por diseño ético.

**Modelos**
- `EmbeddingOnlyBaseline` (Model A solo) vs `FusionModel` (Model A + Model B) vs `StructuredModel` (Model B solo) — mismas regresiones logísticas, distintas columnas de entrada, comparación justa.
- `MockEmbeddingModel`: reemplazo determinístico offline con el mismo interfaz que MiniLM real — permitió construir y probar todo el pipeline sin depender de conexión a Hugging Face.

**Validación del modelo de embeddings**
- STS-Benchmark (1379 pares, similitud humana 0-5): **Spearman r = 0.8203**, dentro del rango publicado para `all-MiniLM-L6-v2` (0.80-0.86). El modelo de embeddings funciona como se espera.

**Evaluación estadística**
- Bootstrap pareado sobre diferencia de AUC y PR-AUC (no solo el punto estimado).
- McNemar's test sobre decisiones binarias pareadas.
- Cross-validation agrupada por wave (GroupKFold, 7 folds) con predicciones out-of-fold sobre el dataset completo, en vez de depender de un único split 75/25.

---

## 2. Resultado actual (Track 1, con embeddings reales)

Este es el hallazgo central hasta ahora, y es genuinamente matizado — no una victoria limpia para la fusión, tampoco un fracaso:

| Métrica | Baseline | Fusión | ¿Diferencia significativa? |
|---|---|---|---|
| AUC (agregado, cross-val) | 0.541 | 0.567 | **Sí**, favorece fusión (CI [0.008, 0.045]) |
| PR-AUC (agregado, cross-val) | 0.203 | 0.208 | No (CI cruza cero) |
| McNemar (decisión binaria a threshold) | — | — | **Sí**, favorece BASELINE (p=2.6e-7) |

**Cómo interpretar esto:** AUC mide calidad de ranking (¿ordena mejor los pares?, sin threshold). McNemar mide decisiones binarias a un threshold específico. Que diverjan sugiere que fusión ordena mejor en general, pero sus probabilidades están calibradas distinto al baseline — predice "match" con más frecuencia, lo que le cuesta accuracy al threshold elegido (base rate global, 0.164) aunque mejore el ranking.

**Conclusión honesta para hoy:** hay evidencia de que la fusión aporta señal de ranking real y estadísticamente significativa sobre el baseline, pero el efecto es modesto y no se traduce limpiamente en mejor clasificación binaria. Esto es un resultado publicable y defendible, pero no el "sí, fusión gana claramente" que uno esperaría idealmente.

## 3. Track 2 (perfiles sintéticos Conecta)

- 20 perfiles hechos a mano (no generados al azar, para que los embeddings tengan contenido semántico real).
- Sin ground truth (no hay `match` real), así que no se entrena un modelo — se usa un score compuesto no supervisado como demostración cualitativa de domain-transfer.
- Con embeddings reales, el ranking tiene sentido semántico (ej. dos estudiantes de derecho/ciencia política interesados en derechos humanos y Model UN quedan arriba del ranking).
- **Pendiente de decisión:** cómo enmarcar esto en el paper — ¿demostración cualitativa explícita, o buscar alguna forma de evaluación cuantitativa?

## 4. Problemas encontrados y resueltos en el camino

Esto vale la pena mencionarlo en el check-in porque muestra rigor metodológico, no solo el resultado final:

1. CSV corrupto → parser custom con recuperación documentada del 99.65%.
2. Filas duplicadas por cita → pseudo-replicación y fuga de datos → corregido con dedupe + split por wave.
3. Threshold fijo 0.5 con clases desbalanceadas (16% match rate) → precision/recall en cero → corregido con threshold basado en base rate + PR-AUC como métrica adicional.
4. `career_c` cargada pero nunca usada → agregada como `same_career`, confirmado que no es redundante con `same_field` (correlación 0.34, moderada).
5. Resultado de un solo split parecía significativo → cross-validation por wave reveló que el hallazgo cambia con más datos agregados (de "no significativo" a "significativo a favor de fusión" en AUC).

## 5. Qué falta para llegar al resultado final

**Ya resuelto (~aprox. 75-80% de la infraestructura técnica):**
- Pipeline de datos, features, modelos, validación estadística — todo construido, probado, y corrido con embeddings reales.
- La pregunta de investigación ya tiene una respuesta preliminar (matizada, no binaria).

**Pendiente:**
- Decidir cómo presentar la divergencia AUC-vs-McNemar en el paper (¿investigar calibración de probabilidades más a fondo, o reportarlo tal cual como hallazgo matizado?).
- Resolver el framing de Track 2 (cualitativo vs. algún tipo de evaluación cuantitativa).
- Documentar explícitamente la limitación de `course_overlap_jaccard` sin proxy real en Track 1.
- Escritura formal del reporte/paper con estos resultados (no iniciada más allá de fixes de formato en el documento anterior).

**Estimación honesta:** la parte de ingeniería y experimentación está mayormente cerrada. Lo que queda es más de análisis/interpretación (decidir cómo contar la historia matizada) y de escritura, no de construir infraestructura nueva.

from typing import Dict, Tuple, Union
import math

def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Devuelve accuracy, precision, recall y f1 dado (tp, fp, fn, tn).
    Si hay divisiones por cero, devuelve float('nan') en esa métrica.
    """
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else float('nan')
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if (not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0)
          else float('nan'))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def from_confusion_matrix(mat_2x2: Tuple[Tuple[int,int], Tuple[int,int]]) -> Dict[str, float]:
    """
    Acepta matriz en formato:
        [[TP, FN],
         [FP, TN]]
    según la convención del PDF: filas = valores reales (Yes/No), columnas = predicción (Yes/No).
    """
    (tp, fn), (fp, tn) = mat_2x2
    return compute_metrics(tp, fp, fn, tn)

# === EJEMPLO con tus 4 matrices ===
# Formato del PDF: filas = reales [Yes, No], columnas = predichos [Yes, No]
M1 = ((95, 7),  (8, 40))
M2 = ((95, 0),  (0, 40))
M3 = ((30, 0),  (40, 0))
M4 = ((500, 0), (45, 0))

for i, M in enumerate([M1, M2, M3, M4], start=1):
    mets = from_confusion_matrix(M)
    print(f"Matriz {i}:")
    for k, v in mets.items():
        print(f"  {k:9s}: {v:.4f}")
    print()

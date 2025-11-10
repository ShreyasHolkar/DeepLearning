from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple

Vector = List[float]

@dataclass
class Model:
    w: Vector
    b: float
    trained: bool
    detail: str

class Perceptron:
    """
    Single-layer perceptron (step activation at 0).
    y = 1 if w·x + b >= 0 else 0
    Update: w += lr*(t - y)*x ; b += lr*(t - y)
    """
    def __init__(self, n_features: int, lr: float = 0.1):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr

    def predict_raw(self, x: Vector) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

    def predict(self, x: Vector) -> int:
        return 1 if self.predict_raw(x) >= 0 else 0

    def fit(self, X: List[Vector], y: List[int], max_epochs: int = 100, shuffle: bool = False) -> Tuple[Vector, float, int]:
        import random
        n = len(X)
        epochs = 0
        while epochs < max_epochs:
            epochs += 1
            idxs = list(range(n))
            if shuffle:
                random.shuffle(idxs)
            updates = 0
            for i in idxs:
                y_hat = self.predict(X[i])
                err = y[i] - y_hat
                if err != 0:
                    # update
                    for k in range(len(self.w)):
                        self.w[k] += self.lr * err * X[i][k]
                    self.b += self.lr * err
                    updates += 1
            if updates == 0:
                break
        return self.w[:], self.b, epochs

def space_2in() -> List[Vector]:
    # Order: (A,B) = (0,0), (0,1), (1,0), (1,1)
    return [[0,0], [0,1], [1,0], [1,1]]

def space_1in() -> List[Vector]:
    # Order: 0, 1
    return [[0], [1]]

TRUTHS: Dict[str, List[int]] = {
    "AND":  [0,0,0,1],
    "OR":   [0,1,1,1],
    "NAND": [1,1,1,0],
    "NOR":  [1,0,0,0],
}
TRUTH_NOT = [1,0]

def build_and() -> Model:
    # Train from scratch (basic training)
    X = space_2in()
    y = TRUTHS["AND"]
    p = Perceptron(n_features=2, lr=0.2)
    w, b, epochs = p.fit(X, y, max_epochs=100, shuffle=False)
    return Model(w=w, b=b, trained=True, detail=f"trained (lr=0.2, epochs={epochs}, no shuffle)")

def build_or() -> Model:
    # Closed-form weights (no training)
    # OR can be separated by w=[1,1], b=-0.5
    return Model(w=[1.0, 1.0], b=-0.5, trained=False, detail="closed-form weights")

def build_nand() -> Model:
    # Train with shuffled data and different lr/epochs
    X = space_2in()
    y = TRUTHS["NAND"]
    p = Perceptron(n_features=2, lr=0.15)
    w, b, epochs = p.fit(X, y, max_epochs=200, shuffle=True)
    return Model(w=w, b=b, trained=True, detail=f"trained (lr=0.15, epochs={epochs}, shuffled)")

def build_nor() -> Model:
    # Closed-form weights (no training)
    # NOR is the complement of OR → w=[-1,-1], b=0.5
    return Model(w=[-1.0, -1.0], b=0.5, trained=False, detail="closed-form weights")

def build_not() -> Model:
    # 1-input perceptron; train it (different from NOR/OR which are closed-form)
    X = space_1in()
    y = TRUTH_NOT
    p = Perceptron(n_features=1, lr=0.2)
    w, b, epochs = p.fit(X, y, max_epochs=20, shuffle=False)
    return Model(w=w, b=b, trained=True, detail=f"trained 1-input (lr=0.2, epochs={epochs})")

def run_gate(name: str):
    name_u = name.strip().upper()
    if name_u == "AND":
        model = build_and()
        X = space_2in()
        truths = TRUTHS["AND"]
        hdr = "(A,B)"
    elif name_u == "OR":
        model = build_or()
        X = space_2in()
        truths = TRUTHS["OR"]
        hdr = "(A,B)"
    elif name_u == "NAND":
        model = build_nand()
        X = space_2in()
        truths = TRUTHS["NAND"]
        hdr = "(A,B)"
    elif name_u == "NOR":
        model = build_nor()
        X = space_2in()
        truths = TRUTHS["NOR"]
        hdr = "(A,B)"
    elif name_u == "NOT":
        model = build_not()
        X = space_1in()
        truths = TRUTH_NOT
        hdr = "A"
    else:
        raise ValueError("Unknown gate. Choose from: AND, OR, NAND, NOR, NOT")

    def predict(x: Vector) -> int:
        s = sum(wi*xi for wi, xi in zip(model.w, x)) + model.b
        return 1 if s >= 0 else 0

    print(f"\nGate: {name_u}")
    print(f"Method: {model.detail}")
    print(f"Weights: {model.w} | Bias: {model.b:.3f}")
    print(f"\nTruth table {hdr} -> (target | perceptron):")
    for x, t in zip(X, truths):
        y_hat = predict(x)
        print(f"  {tuple(x)} -> ({t} | {y_hat})")

    correct = all(predict(x) == t for x, t in zip(X, truths))
    print("\nResult:", "✔ Perceptron matches the gate." if correct else "✘ Mismatch detected (should not happen for these gates).")

if __name__ == "__main__":
    try:
        choice = input("which gate you want to implement: ").strip()
        run_gate(choice)
    except Exception as e:
        print("Error:", e)

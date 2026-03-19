"""
VQR - Variational Quantum Regressor
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector, Pauli
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_LAYERS = 2
MAXITER = 200
NUM_OBS = 3


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def build_observables():
    obs_list = []
    for i in range(NUM_QUBITS):
        label = ['I'] * NUM_QUBITS
        label[i] = 'Z'
        obs_list.append(Pauli(''.join(label)))
    return obs_list[:NUM_OBS]


class VQRegressor:
    def __init__(self):
        self.feature_map = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)
        self.ansatz = TwoLocal(
            num_qubits=NUM_QUBITS,
            rotation_blocks='ry',
            entanglement_blocks='cz',
            entanglement='linear',
            reps=NUM_LAYERS
        )
        self.observables = build_observables()
        self.circuit = self.feature_map.compose(self.ansatz)
        self.num_ansatz_params = len(self.ansatz.parameters)
        self.weights = np.zeros(NUM_OBS, dtype=float)
        self.bias = 0.0
        self.theta = np.zeros(self.num_ansatz_params, dtype=float)

    def _expectation_values(self, x, theta):
        param_dict = {}
        for p, val in zip(self.feature_map.parameters, x):
            param_dict[p] = float(val)
        for p, val in zip(self.ansatz.parameters, theta):
            param_dict[p] = float(val)
        bound = self.circuit.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound)
        return np.array([
            float(sv.expectation_value(obs).real)
            for obs in self.observables
        ])

    def _predict_single(self, x, theta, weights, bias):
        evs = self._expectation_values(x, theta)
        return float(evs @ weights + bias)

    def predict(self, X):
        return np.array([
            self._predict_single(x, self.theta, self.weights, self.bias)
            for x in X
        ])

    def fit(self, X, y):
        n_total = self.num_ansatz_params + NUM_OBS + 1

        def loss(params):
            th = params[:self.num_ansatz_params]
            w = params[self.num_ansatz_params:self.num_ansatz_params + NUM_OBS]
            b = params[-1]
            preds = np.array([
                self._predict_single(x, th, w, b) for x in X
            ])
            return float(np.mean((preds - y) ** 2))

        x0 = np.concatenate([
            self.theta,
            np.random.uniform(-0.1, 0.1, NUM_OBS),
            [0.0]
        ])

        res = scipy_minimize(loss, x0, method='COBYLA',
                             options={'maxiter': MAXITER, 'rhobeg': 0.3})

        self.theta = res.x[:self.num_ansatz_params]
        self.weights = res.x[self.num_ansatz_params:
                             self.num_ansatz_params + NUM_OBS]
        self.bias = res.x[-1]
        return res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_all = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- VQR ({NUM_QUBITS}q, {NUM_OBS} obs, "
          f"COBYLA {MAXITER} iter) ---")
    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        y = build_empirical(draws, pos)

        vqr = VQRegressor()
        final_loss = vqr.fit(X_all, y)

        pred = vqr.predict(X_all)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"loss={final_loss:.6f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (VQR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- VQR (5q, 3 obs, COBYLA 200 iter) ---
  Poz 1... loss=0.001904  top: 7:0.063 | 2:0.056 | 28:0.048
  Poz 2... loss=0.000710  top: 10:0.056 | 26:0.051 | 8:0.051
  Poz 3... loss=0.000448  top: 11:0.094 | 14:0.060 | 13:0.059
  Poz 4... loss=0.000437  top: 23:0.080 | 14:0.066 | 26:0.054
  Poz 5... loss=0.000542  top: 24:0.071 | 11:0.061 | 23:0.058
  Poz 6... loss=0.000902  top: 24:0.075 | 12:0.071 | 25:0.065
  Poz 7... loss=0.002423  top: 15:0.087 | 38:0.062 | 18:0.058

==================================================
Predikcija (VQR, deterministicki, seed=39):
[7, 10, x, y, z, 25, 38]
==================================================
"""



"""
VQR - Variational Quantum Regressor

Multi-observable pristup: meri 3 razlicita Pauli operatora (Z na razlicitim qubitima)
Izlaz = linearna kombinacija ekspektacionih vrednosti: y = w1*<Z_0> + w2*<Z_1> + w3*<Z_2> + bias
COBYLA optimizuje i parametre kola i tezine/bias zajedno
Bogatiji model od q_3 (koji koristi samo jedan Z^5 observable)
Egzaktno, deterministicki, Statevector za ekspektacione vrednosti
"""

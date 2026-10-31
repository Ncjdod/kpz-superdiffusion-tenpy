"""Open spin-1/2 XXZ chain, H/J = -sum_i (SxSx + SySy + Delta SzSz), evolved with TeNPy TEBD."""

import numpy as np
from scipy.linalg import eigh
from tenpy.algorithms.purification import PurificationTEBD
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.linalg import np_conserved as npc
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.networks.purification_mps import PurificationMPS

from kpz import DELTA


def make_model(length, delta=DELTA):
    """Ferromagnetic XXZ chain with open ends and Sz conservation, in units of J."""
    return SpinChain({"L": length, "S": .5, "Jx": -1., "Jy": -1., "Jz": -float(delta),
                      "conserve": "Sz", "bc_MPS": "finite", "bc_x": "open"})


def initial_wall(model, eta):
    """Product state with 2<Sz> = +eta on the left half and -eta on the right, purified unless eta = 1."""
    sites = model.lat.mps_sites()
    length = len(sites)
    if eta == 1:
        return MPS.from_product_state(sites, ["up"] * (length // 2) + ["down"] * (length // 2),
                                      bc="finite", unit_cell_width=length)
    tensors = []
    for index, site in enumerate(sites):
        signed_eta = eta if index < length // 2 else -eta
        weights = np.empty(site.dim)
        weights[site.state_index("up")] = (1 + signed_eta) / 2
        weights[site.state_index("down")] = (1 - signed_eta) / 2
        tensor = npc.diag(np.sqrt(weights), site.leg, labels=["p", "q"])
        tensor = tensor.add_trivial_leg(0, label="vL", qconj=1)
        tensor = tensor.add_trivial_leg(1, label="vR", qconj=-1)
        tensors.append(tensor)
    return PurificationMPS(sites, tensors, [[1.]] * (length + 1), bc="finite",
                           form="B", unit_cell_width=length)


def make_engine(state, model, eta, dt, chi, steps, resume=None):
    """Second-order TEBD; purified states use backwards disentangling."""
    options = {"dt": dt, "N_steps": steps, "order": 2,
               "trunc_params": {"chi_max": chi, "svd_min": 1.e-12}}
    engine_type = TEBDEngine
    if eta != 1:
        engine_type = PurificationTEBD
        options["disentangle"] = "backwards"
    if resume is None:
        return engine_type(state, model, options)
    options["start_trunc_err"] = resume["truncation"]
    return engine_type(state, model, options, resume_data=resume["engine"])


def measure(state, model, engine):
    """Observables of a canonical copy, so that the truncation gauge does not enter."""
    raw = state.expectation_value("Sz")
    snapshot = state.copy()
    snapshot.canonical_form()
    sz = np.real(snapshot.expectation_value("Sz"))
    return {"u": float(np.real(engine.evolved_time)), "sz": sz,
            "energy": float(np.real(np.sum(model.bond_energies(snapshot)))),
            "entropy": float(snapshot.entanglement_entropy(bonds=[state.L // 2])[0]),
            "chi": max(state.chi), "discarded": float(engine.trunc_err.eps),
            "canonical_change": float(np.max(np.abs(sz - raw)))}


def evolve(length, eta, dt, chi, times, delta=DELTA):
    """Sz profiles at uniformly spaced times starting at zero."""
    model = make_model(length, delta)
    state = initial_wall(model, eta)
    engine = make_engine(state, model, eta, dt, chi, round((times[1] - times[0]) / dt))
    profiles = [measure(state, model, engine)["sz"]]
    for _ in times[1:]:
        engine.run()
        profiles.append(measure(state, model, engine)["sz"])
    return np.asarray(profiles)


def exact_wall(length, times, eta, delta=DELTA):
    """Sz profiles of the same mixed wall from full diagonalisation of the 2**L Hamiltonian."""
    site = make_model(length, delta).lat.mps_sites()[0]
    sp, sm, sz = [site.get_op(name).to_ndarray() for name in ("Sp", "Sm", "Sz")]
    sx, sy = (sp + sm) / 2, (sp - sm) / 2j

    def embed(operators):
        matrix = np.array([[1.]])
        for operator in operators:
            matrix = np.kron(matrix, operator)
        return matrix

    hamiltonian = np.zeros((2 ** length, 2 ** length), dtype=complex)
    for index in range(length - 1):
        for operator, coefficient in ((sx, -1), (sy, -1), (sz, -delta)):
            operators = [np.eye(2)] * length
            operators[index] = operators[index + 1] = operator
            hamiltonian += coefficient * embed(operators)
    probability = np.ones(1)
    observables = []
    for index in range(length):
        signed_eta = eta if index < length // 2 else -eta
        local = np.empty(2)
        local[site.state_index("up")] = (1 + signed_eta) / 2
        local[site.state_index("down")] = (1 - signed_eta) / 2
        probability = np.kron(probability, local)
        operators = [np.eye(2)] * length
        operators[index] = sz
        observables.append(np.diag(embed(operators)).real)
    energies, vectors = eigh(hamiltonian)
    profiles = []
    for time in times:
        unitary = (vectors * np.exp(-1j * time * energies)) @ vectors.conj().T
        profiles.append(np.asarray(observables) @ (np.abs(unitary) ** 2 @ probability))
    return np.asarray(profiles)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:51:43 2026

@author: ckadelka
"""

import numpy as np
from typing import Tuple
import boolforge

# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

_LOGIC_MAP = {
    "AND": "&",
    "and": "&",
    "&&": "&",
    "&": "&",
    "OR": "|",
    "or": "|",
    "||": "|",
    "|": "|",
    "NOT": "~",
    "not": "~",
    "!": "~",
    "~": "~",
}


_COMPARE_OPS = {"==", "!=", ">=", "<=", ">", "<"}

_ARITH_OPS = {"+", "-", "*", "%"}


def _is_number(token: str) -> bool:
    """Return True if token is a pure numeric literal."""
    try:
        float(token)
        return True
    except ValueError:
        return False


# --------------------------------------------------
# Main function
# --------------------------------------------------


def f_from_expression(
    expr: str,
    max_degree: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Boolean truth table from a biological Boolean expression.

    Robust against:
        - Unicode names
        - Names containing /, -, +
        - Leading digits (e.g. 3oxo)
        - Arithmetic operators
        - Comparison operators
    """

    # --------------------------------------------------
    # 1. Normalize parentheses spacing
    # --------------------------------------------------

    expr = expr.replace("(", " ( ").replace(")", " ) ")

    raw_tokens = expr.split()

    tokens = []
    variables = []
    seen = set()

    # --------------------------------------------------
    # 2. Token classification
    # --------------------------------------------------

    for token in raw_tokens:

        if token in {"(", ")"}:
            tokens.append(token)
            continue

        if token in _LOGIC_MAP:
            tokens.append(_LOGIC_MAP[token])
            continue

        if token in _COMPARE_OPS:
            tokens.append(token)
            continue
        
        if token in _ARITH_OPS:
            tokens.append(token)
            continue

        if _is_number(token):
            tokens.append(token)
            continue

        # Otherwise: biological identifier
        if token not in seen:
            seen.add(token)
            variables.append(token)

        tokens.append(token)

    n = len(variables)

    if n > max_degree:
        return np.array([], dtype=np.uint8), np.array(variables)

    # --------------------------------------------------
    # 3. Map biological names → safe Python names
    # --------------------------------------------------

    safe_map = {var: f"v{i}" for i, var in enumerate(variables)}

    safe_tokens = [
        safe_map[token] if token in safe_map else token
        for token in tokens
    ]

    expr_mod = " ".join(safe_tokens)

    # --------------------------------------------------
    # 4. Build evaluation environment
    # --------------------------------------------------

    truth_table = boolforge.get_left_side_of_truth_table(n)

    local_dict = {
        safe_map[var]: truth_table[:, i].astype(np.int64)
        for i, var in enumerate(variables)
    }

    # --------------------------------------------------
    # 5. Evaluate expression
    # --------------------------------------------------

    try:
        result = eval(expr_mod, {"__builtins__": None}, local_dict)
    except Exception as e:
        raise ValueError(
            f"Error evaluating expression:\n{expr}\nParsed as:\n{expr_mod}\nError: {e}"
        )

    # --------------------------------------------------
    # 6. Enforce Boolean semantics
    # --------------------------------------------------

    result = np.asarray(result)

    if n == 0:
        result = np.array([int(result)], dtype=np.int64)
    else:
        result = result.astype(np.int64)

    # Fix NOT and enforce {0,1}
    result = result & 1

    return result.astype(np.uint8), np.array(variables)

def from_string(
    network_string: str,
    separator: str = "=",
    max_degree: int = 24,
    allow_truncation: bool = False,
    simplify_functions: bool = False,
):

    # --------------------------------------------
    # 1. Clean lines
    # --------------------------------------------
    lines = [
        l.strip()
        for l in network_string.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]

    rules = []
    for i, line in enumerate(lines):
        if separator not in line:
            raise ValueError(f"Missing separator '{separator}' in line {i+1}:\n{line}")
        lhs, rhs = line.split(separator, 1)
        rules.append((lhs.strip(), rhs.strip()))

    # --------------------------------------------
    # 2. Collect explicitly defined nodes
    # --------------------------------------------
    node_names = [lhs for lhs, _ in rules]

    # --------------------------------------------
    # 3. Parse RHS to detect all regulators
    # --------------------------------------------
    parsed_rhs = []
    all_regulators = set()

    for lhs, rhs in rules:
        f, regulators = f_from_expression(rhs, max_degree=max_degree)
        parsed_rhs.append((lhs, rhs, f, regulators))
        for r in regulators:
            all_regulators.add(r)

    # --------------------------------------------
    # 4. Add missing regulators as identity nodes
    # --------------------------------------------
    missing_nodes = sorted(all_regulators - set(node_names))

    # Append them deterministically (sorted for reproducibility)
    node_names_extended = node_names + missing_nodes

    node_index = {name: i for i, name in enumerate(node_names_extended)}

    # --------------------------------------------
    # 5. Build F and I
    # --------------------------------------------
    F = []
    I = []

    # First build defined rules
    for lhs, rhs, f, regulators in parsed_rhs:

        deg = len(regulators)

        if deg > max_degree:
            if not allow_truncation:
                raise ValueError(
                    f"Node '{lhs}' has indegree {deg} > max_degree={max_degree}."
                )
            idx = node_index[lhs]
            F.append(np.array([0, 1], dtype=int))
            I.append(np.array([idx], dtype=int))
            continue

        reg_indices = [node_index[r] for r in regulators]

        F.append(f.astype(int))
        I.append(np.array(reg_indices, dtype=int))

    # Then add identity nodes for missing regulators
    for name in missing_nodes:
        idx = node_index[name]
        F.append(np.array([0, 1], dtype=int))  # identity
        I.append(np.array([idx], dtype=int))

    return boolforge.BooleanNetwork(
        F,
        I,
        node_names_extended,
        simplify_functions=simplify_functions,
    )






from boolforge.bio_models import fetch_file

dummy = boolforge.get_bio_models_from_repository()
bns_true = dummy['BooleanNetworks']


download_url_base = (
    'https://raw.githubusercontent.com/ckadelka/'
    'DesignPrinciplesGeneNetworks/main/'
    'update_rules_122_models_Kadelka_SciAdv/'
)
download_url = download_url_base + 'all_txt_files.csv'
csv = fetch_file(download_url)


bns = []
successful_download_urls = []
failed_download_urls = []

for line in csv.splitlines():
    download_url = download_url_base + line
    if '.txt' in download_url:
        try:
            if 'tabular' in download_url:
                pass
            else:
                string = fetch_file(download_url)
                bn = from_string(
                    string,
                    separator = '=',
                    simplify_functions=False,
                )

            successful_download_urls.append(download_url)
            bns.append(bn)

        except Exception:
            failed_download_urls.append(download_url)

j=0
for i in range(bns_true[j].N):
    print(bns[j].I[i], bns_true[j].I[i])
    print(bns[j].F[i], bns_true[j].F[i])
    print()
    
    
for download_url in failed_download_urls:
    string = fetch_file(download_url)
    bn = from_string(
        string,
        separator = '=',
        simplify_functions=False,
    )    

#variables are the same
for i in range(122):
    try:
        print(i,set(bns[i].variables) == set(bns_true[i].variables))
    except ValueError:
        print(i,'failed')

#but the variable order is often not the same
for i in range(122):
    try:
        print(i,np.all((bns[i].variables) == (bns_true[i].variables)))
    except ValueError:
        print(i,'failed')
        
#fix variable order and compare I
for i in range(122):
    try:
        bn = bns[i]
        bn_true = bns_true[i]

        # Map variable name -> index for each BN
        name_to_index = {v: k for k, v in enumerate(bn.variables)}
        name_to_index_true = {v: k for k, v in enumerate(bn_true.variables)}

        comparisons = []

        for var in bn.variables:
            # regulators in new BN (by name)
            idx = name_to_index[var]
            regs = {bn.variables[k] for k in bn.I[idx]}

            # regulators in old BN (by name)
            idx_true = name_to_index_true[var]
            regs_true = {bn_true.variables[k] for k in bn_true.I[idx_true]}

            comparisons.append(regs == regs_true)

        print(i, np.mean(comparisons))

    except Exception:
        print(i, "failed")

#same Hamming weight for every function
for i in range(122):
    try:
        print(i,np.mean([sum(f)==sum(g) for f,g in zip(bns[i].F,bns_true[i].F)]))
    except ValueError:
        print(i,'failed')

#same exact function for every function
def functions_equal_modulo_reg_order(bn, bn_true):
    """
    Returns True if all node functions are identical
    modulo regulator ordering.
    """

    # map variable name -> index
    idx_bn = {v: i for i, v in enumerate(bn.variables)}
    idx_true = {v: i for i, v in enumerate(bn_true.variables)}

    for var in bn.variables:

        i1 = idx_bn[var]
        i2 = idx_true[var]

        regs1 = [bn.variables[k] for k in bn.I[i1]]
        regs2 = [bn_true.variables[k] for k in bn_true.I[i2]]

        if set(regs1) != set(regs2):
            return False  # regulators differ

        f1 = bn.F[i1].f
        f2 = bn_true.F[i2].f

        k = len(regs1)

        if k <= 1:
            if not np.array_equal(f1, f2):
                return False
            continue

        # compute permutation mapping regs1 order -> regs2 order
        perm = [regs1.index(r) for r in regs2]

        # reshape f1 to tensor
        f1_tensor = f1.reshape([2]*k)

        # permute axes
        f1_perm = np.transpose(f1_tensor, axes=perm).flatten()

        if not np.array_equal(f1_perm, f2):
            return False

    return True

for i in range(122):
    try:
        equal = functions_equal_modulo_reg_order(bns[i], bns_true[i])
        print(i, equal)
    except Exception:
        print(i, "failed")
        
for i,line in enumerate(csv.splitlines()):
    print(i,'tabular' in line)

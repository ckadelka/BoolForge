#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:54:16 2026

@author: ckadelka
"""

import numpy as np

from ..backend._numba import __LOADED_NUMBA__, _numba_required
if __LOADED_NUMBA__:
    from ..backend.robustness_async import _compute_local_coherence_async_numba

def get_trap_space_dimension(points):
    ref = points[0]
    varying = 0
    for s in points[1:]:
        varying |= (ref ^ s)
    return varying.bit_count() 

class BooleanNetworkRobustnessAsyncMixin:
    def get_terminal_sccs_and_robustness_asynchronous_exact(self) -> dict:
        """
        Compute terminal SCCs (attractors) and exact robustness measures of an 
        asynchronously updated Boolean network.

        This method constructs the exact asynchronous state transition graph
        on ``2**N`` states and interprets the asynchronous dynamics as a finite 
        Markov chain. All attractors (terminal SCCs), basin sizes, and the 
        terminal SCCs reached  from each state are determined exactly. Based on
        this decomposition, exact coherence measures are computed for the full
        network, for each basin of attraction, and for each attractor.

        This computation requires memory and time proportional to ``2**N`` and
        is intended for small-to-moderate networks (e.g., ``N ≈ 18`` on typical 
        hardware).

        Returns
        -------
        dict
            Dictionary with the following keys:

            - TerminalSCCs : list[list[int]]
                Each terminal SCC represented as a recurrent communicating class.
            - NumberOfTerminalSCCs : int
                Total number of terminal SCCs.
            - LengthOfTerminalSCCs : np.ndarray of int
                Length of each terminal SCC.
            - TrapSpaceDimensions : np.ndarray of int
                Dimension of the minimal space containing each terminal SCC.
            - BasinSizes : np.ndarray of float
                Probability of reaching a terminal SCC from a random state.
            - AbsorptionProbabilities : np.ndarray of float
                For each of the ``2**N`` states, the probability that a specific
                terminal SCC is reached
            - Coherence : float
                Exact global network coherence.
            - BasinCoherences : np.ndarray of float
                Exact coherence of each basin of attraction.
            - TerminalSCCCoherences : np.ndarray of float
                Exact coherence of each terminal SCC.
        """
        if not __LOADED_NUMBA__:
            _numba_required("Asynchronous exact robustness computation")
        
        terminal_sccs = self.get_terminal_sccs_asynchronous_exact()
        absorption_probs = self.get_absorption_probabilities_exact()
        
        basin_sizes = absorption_probs.sum(axis=0)
        length_terminal_sccs = np.array(list(map(len,terminal_sccs)))
        dim_trap_spaces = np.array(list(map(get_trap_space_dimension,terminal_sccs)))
        local_coherence = _compute_local_coherence_async_numba(
            self.N,
            absorption_probs
        )
        basin_coherences = (absorption_probs.T @ local_coherence) / basin_sizes
        coherence = np.dot(basin_coherences, basin_sizes) / float(1<<self.N)
        terminal_scc_coherences = np.array([
            np.mean(local_coherence[a]) for a in terminal_sccs
        ])
        
        return  {
            "TerminalSCCs": terminal_sccs,
            "NumberOfTerminalSCCs": int(len(terminal_sccs)),
            "LengthOfTerminalSCCs": length_terminal_sccs,
            "TrapSpaceDimensions": dim_trap_spaces,
            "BasinSizes": basin_sizes / float(1<<self.N),
            "AbsorptionProbabilities": absorption_probs,
            "Coherence": coherence.item(),
            "BasinCoherences": basin_coherences,
            "TerminalSCCCoherences": terminal_scc_coherences,
        }        

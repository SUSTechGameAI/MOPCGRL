import math

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from pymoo.decomposition.asf import ASF
from pymoo.decomposition.pbi import PBI
from pymoo.util.dominator import Dominator
from pymoo.util.function_loader import load_function
from pymoo.util.misc import random_permuations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import copy

# =========================================================================================================
# Implementation
# Following original code by K. Li https://cola-laboratory.github.io/codes/CTAEA.zip
# =========================================================================================================

def constr_to_cv(G, constr_p):
    CV = []
    for g in G:
        alpha = g/constr_p - 1
        if alpha >= 0:
            CV.append(0)
        else:
            CV.append(-alpha)
    return CV


def merge(a, b):
    if a is None:
        return b
    elif b is None:
        return a

    if len(a) == 0:
        return b
    elif len(b) == 0:
        return a
    else:
        obj = np.concatenate([a, b])
        return obj

# input: Hm(CA and DA), P(index of parents)
# output: S(chosen indices, array shape (n,1))
def comp_by_cv_dom_then_random(pop, P):
    S = np.full(P.shape[0], np.nan)
    
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        
        # get F and CV from a and b
        F_a, CV_a = pop[a][1], pop[a][2]
        F_b, CV_b = pop[b][1], pop[b][2]

        if CV_a <= 0.0 and CV_b <= 0.0:
            rel = Dominator.get_relation(F_a, F_b)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b
            else:
                S[i] = np.random.choice([a, b])
        elif CV_a <= 0.0:
            S[i] = a
        elif CV_b <= 0.0:
            S[i] = b
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int)


class RestrictedMating:
    def __init__(self):
        self.pressure = 2
    
    """Restricted mating approach to balance convergence and diversity archives"""
    # Algorithm 4: Restricted Mating Selection
    # input: Combined array of CA and DA, n_select(pop_size), n_parents
    # output: mating parents p1, p2
    def _do(self, Hm, n_select, n_parents):
        n_pop = len(Hm) // 2
        F = np.vstack(Hm[:, 1])
        _, rank = NonDominatedSorting().do(F, return_rank=True)
        # proportion of nondominated solution of CA and DA
        Pc = (rank[:n_pop] == 0).sum() / len(Hm)
        Pd = (rank[n_pop:] == 0).sum() / len(Hm)

        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure
        n_perms = math.ceil(n_random / n_pop)
        # get random permutations and reshape them
        P = random_permuations(n_perms, n_pop)[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))
        if Pc <= Pd:
            # Choose from DA
            P[::n_parents, :] += n_pop
        pf = np.random.random(n_select)
        P[1::n_parents, :][pf >= Pc] += n_pop

        # compare using tournament function
        S = comp_by_cv_dom_then_random(Hm, P)
        
        return np.reshape(S, (n_select, n_parents))


class CADASurvival:

    def __init__(self, ref_dirs):
        self.ref_dirs = ref_dirs
        self.opt = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self._decomposition = ASF()
        # self._decomposition = PBI(theta=5)
        self._calc_perpendicular_distance = load_function("calc_perpendicular_distance")


    def do(self, ca, da, off, sac_population, n_survive=None):
        # combine every model with its F and CV. The structure of the array: [sac_model, F, CV]
        if len(da) == 0:
            self.ideal_point = np.min(np.vstack(ca[:,1]), axis=0)
            return ca, copy.deepcopy(ca)
        else:
            # Offspring are last of merged population    
            F_off = np.vstack(off[:, 1]) 
     
            # Update ideal point. off.get → individual.get
            self.ideal_point = np.min(np.vstack((self.ideal_point, F_off)), axis=0)

            # Update CA
            ca_off = merge(ca, off)
            ca_pop = self._updateCA(ca_off, n_survive, sac_population)
            
            # Update DA
            da_off = merge(da, off)
            da_pop = self._updateDA(ca_pop, da_off, n_survive, sac_population)
            
            return ca_pop, da_pop

    # Algorithm1: Association Procedure
    # input: self(CADAsurvival: ideal point, reference direction or weight set W in paper), pop
    # output: a list of niches, a list of decomposed fitness matrix (size: pop_size)
    def _associate(self, pop, sac_population):
        """Associate each individual with a F vector and calculate decomposed fitness"""
        # F = pop.get("F")
        F = np.vstack(pop[:, 1])
        # CTAEA paper Algorithm1: line 2-6
        dist_matrix = self._calc_perpendicular_distance(F - self.ideal_point, self.ref_dirs)
        niche_of_individuals = np.argmin(dist_matrix, axis=1)
        # decomposed fitness matrix
        FV = self._decomposition.do(F, weights=self.ref_dirs[niche_of_individuals, :],
                                    ideal_point=self.ideal_point, weight_0=1e-4)
        # pop.set → ind.set
        sac_population.niche = niche_of_individuals
        sac_population.FV = FV
        return niche_of_individuals, FV

    # Algorithm2: Update Mechanism of CA
    # input: self(CADAsurvival: ideal point, reference direction or weight set W in paper), pop, n_survive(pop_size)
    # output: S(a list of survived individuals)
    def _updateCA(self, sac_pop, n_survive, sac_population):
        """Update the Convergence archive (CA)"""
        # CTAEA paper: line 3-4
        CV = np.array(sac_pop[:, 2])  # Extract CV values
        Sc = sac_pop[CV == 0]  # ConstraintsAsObjective population, this list is empty in the beginning
        # CTAEA paper: line 5-6
        if len(Sc) == n_survive:  # Exactly n_survive feasible individuals
            F = np.vstack(Sc[:, 1])  # Extract F values
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            sac_population.rank = rank
            self.opt = Sc[fronts[0]]
            return Sc
        elif len(Sc) < n_survive:  # Not enough feasible individuals
            # CTAEA paper: line 23-25
            remainder = n_survive - len(Sc)
            # Solve sub-problem CV, tche
            # pick out the individuals that violated the constraint
            SI = sac_pop[CV > 0]
            f1 = np.array(SI[:, 2])
            _, f2 = self._associate(SI, sac_population)
            sub_F = np.column_stack([f1, f2])
            fronts = NonDominatedSorting().do(sub_F, n_stop_if_ranked=remainder)
            I = []
            for front in fronts:
                # CTAEA paper: line 26-27
                if len(I) + len(front) <= remainder:
                    I.extend(front)
                # CTAEA paper: line 28-29
                else:
                    n_missing = remainder - len(I)
                    # sort by CV
                    last_front_CV = np.argsort(f1.flatten()[front])
                    # pick solutions with smaller CV
                    I.extend(front[last_front_CV[:n_missing]])
            SI = SI[I]
            # combine feasible and infeasible ind
            S = merge(Sc, SI)
            F = np.vstack(S[:, 1])
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            sac_population.rank = rank
            self.opt = S[fronts[0]]
            return S
        
        else:  # Too many feasible individuals
            # CTAEA paper: line 7-8
            F = np.vstack(Sc[:, 1])
            # Filter by non-dominated sorting
            fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
            # CTAEA paper: line 9-10
            I = np.concatenate(fronts)
            S, rank, F = Sc[I], rank[I], F[I]
            # CTAEA paper: line 11-21
            if len(S) > n_survive:
                # CTAEA paper: line 11-14
                # Remove individual in most crowded niche and with worst fitness
                niche_of_individuals, FV = self._associate(S, sac_population)
                index, count = np.unique(niche_of_individuals, return_counts=True)
                survivors = np.full(S.shape[0], True)
                # CTAEA paper: line 15-16
                while survivors.sum() > n_survive:
                    # find the most crowded niches
                    crowdest_niches, = np.where(count == count.max())
                    worst_idx = None
                    worst_niche = None
                    worst_fit = -1
                    # CTAEA paper: line 17-21
                    for crowdest_niche in crowdest_niches:
                        crowdest, = np.where((niche_of_individuals == index[crowdest_niche]) & survivors)
                        niche_worst = crowdest[FV[crowdest].argmax()]
                        dist_to_max_fit = cdist(F[[niche_worst], :], F).flatten()
                        dist_to_max_fit[niche_worst] = np.inf
                        dist_to_max_fit[~survivors] = np.inf
                        min_d_to_max_fit = dist_to_max_fit.min()

                        dist_in_niche = squareform(pdist(F[crowdest]))
                        np.fill_diagonal(dist_in_niche, np.inf)

                        delta_d = dist_in_niche - min_d_to_max_fit
                        min_d_i = np.unravel_index(np.argmin(delta_d, axis=None), dist_in_niche.shape)
                        if (delta_d[min_d_i] < 0) or (
                                delta_d[min_d_i] == 0 and (FV[crowdest[list(min_d_i)]] > niche_worst).any()):
                            min_d_i = list(min_d_i)
                            np.random.shuffle(min_d_i)
                            closest = crowdest[min_d_i]
                            niche_worst = closest[np.argmax(FV[closest])]
                        if (FV[niche_worst] > worst_fit).all():
                            worst_fit = FV[niche_worst]
                            worst_idx = niche_worst
                            worst_niche = crowdest_niche
                    survivors[worst_idx] = False
                    count[worst_niche] -= 1
                S, rank = S[survivors], rank[survivors]
            sac_population.rank = rank
            self.opt = S[rank == 0]
            return S

    # Algorithm 3: Update Mechanism of the DA. Take CA as a reference, consider the under-exploited areas
    # input: self, CA pop, merged pop of diversity archive(start with an empty list) and offspring pop, n_survive(pop_size)
    # output: DA pop
    def _updateDA(self, ca_pop, Hd, n_survive, sac_population):
        """Update the Diversity archive (DA)"""
        # pseudo-code line 1-4 
        niche_Hd, FV = self._associate(Hd, sac_population)
        niche_CA, _ = self._associate(ca_pop, sac_population)
        itr = 1
        S = []
        count = 0
        # pseudo-code line 5-15
        while len(S) < n_survive:
            # print(count)
            count += 1
            for i in range(n_survive):
                # solutions in CA survive
                current_ca, = np.where(niche_CA == i)
                if len(current_ca) < itr:
                    # line 8-12
                    for _ in range(itr - len(current_ca)):
                        current_da = np.where(niche_Hd == i)[0]
                        if current_da.size > 0:
                            # the best nondominated solutions in DA is added
                            F = np.vstack(Hd[current_da][:, 1])
                            nd = NonDominatedSorting().do(F, only_non_dominated_front=True, n_stop_if_ranked=0)
                            i_best = current_da[nd[np.argmin(FV[current_da[nd]])]]
                            niche_Hd[i_best] = -1
                            if len(S) < n_survive:
                                S.append(i_best)
                        else:
                            break
                if len(S) == n_survive:
                    break
            itr += 1
        return Hd[S]

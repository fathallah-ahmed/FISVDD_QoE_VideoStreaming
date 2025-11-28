import numpy as np
from scipy.spatial import distance
import time
import pandas as pd
from typing import Tuple, List, Optional

class fisvdd:
    """
    Fast Incremental Support Vector Data Description (FISVDD).
    """
    def __init__(self, data: np.ndarray, sigma: float, eps_cp: float = 1e-8, eps_ol: float = 1e-8):
        """
        Initialize the FISVDD model.

        Args:
            data (np.ndarray): Initial dataset to initialize the model.
            sigma (float): Gaussian kernel bandwidth.
            eps_cp (float): Epsilon for close points (numerical stability).
            eps_ol (float): Epsilon for outliers (numerical stability).
        """
        self.data = data
        self.sigma = sigma
        self.eps_cp = eps_cp
        self.eps_ol = eps_ol

        # Initialize with the first data point
        self.inv_A = np.array([[1.0]])
        self.alpha = np.array([1.0])
        self.sv = np.array([self.data[0]])
        self.obj_val: List[float] = []
        self.score = 1.0

    def _print_res(self):
        print("\nalpha -------")
        print(self.alpha)
        print("\nsupport vector -------")
        print(self.sv)

    def find_sv(self):
        """
        Train the FISVDD model on the initial data provided in __init__.
        Iterates through the data points and updates the support vectors.
        """
        for new_data in self.data[1:]:
            new_data = np.array([new_data])

            score, sim_vec = self.score_fcn(new_data)
            if score > 0:
                self.expand(new_data, sim_vec)

                if min(self.alpha) < 0:
                    backup = self.shrink()
                    for each in backup:
                        each = np.array([each])
                        score, sim_vec = self.score_fcn(each)
                        if score > 0:
                            self.expand(each, sim_vec)

                self.model_update()

            self.obj_val.append(self.score)

    def up_inv(self, prev_inv: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Calculate the inverse of A_(k+1) based on Lemma 2 (Sherman-Morrison-Woodbury-like update).

        Args:
            prev_inv (np.ndarray): Inverse of A_k.
            v (np.ndarray): Similarity vector between new data point and support vectors.

        Returns:
            np.ndarray: Inverse of A_(k+1).
        """
        p = np.dot(prev_inv, v)
        beta = 1 - np.dot(v, p)
        # Avoid division by zero if beta is extremely small
        if abs(beta) < 1e-12:
             beta = 1e-12

        A = prev_inv + np.outer(p, p) / beta
        C = - p / beta
        C = C.reshape(1, -1) # Ensure C is a row vector (1, N)
        B = np.reshape(- p / beta, (len(p), 1)) # B is column vector (N, 1)
        D = np.array([[1 / beta]])
        res = np.vstack((np.hstack((A, B)), np.hstack((C, D))))
        return res

    def down_inv(self, next_inv: np.ndarray) -> np.ndarray:
        """
        Calculate the inverse of A_k based on Lemma 3 (Downdating).

        Args:
            next_inv (np.ndarray): Inverse of A_(k+1).

        Returns:
            np.ndarray: Inverse of A_k.
        """
        lamb = next_inv[-1, -1]
        if abs(lamb) < 1e-12:
            lamb = 1e-12
            
        u = next_inv[:-1, -1]
        res = next_inv[:-1, :-1] - np.outer(u, u) / lamb
        return res

    def expand(self, new_sv: np.ndarray, new_sim_vec: np.ndarray):
        """
        Expand the support vector set (Algorithm 1).

        Args:
            new_sv (np.ndarray): The new support vector to add.
            new_sim_vec (np.ndarray): Similarity vector.
        """
        self.inv_A = self.up_inv(self.inv_A, new_sim_vec)
        self.alpha = np.sum(self.inv_A, axis=1)
        self.sv = np.vstack((self.sv, new_sv))

    def shrink(self) -> List[np.ndarray]:
        """
        Shrink the support vector set (Algorithm 2).
        Removes support vectors with negative alpha values.

        Returns:
            List[np.ndarray]: List of removed support vectors (backup).
        """
        backup = []
        while True:
            # Find index of minimum alpha
            min_ind = np.argmin(self.alpha)
            
            # If the minimum alpha is non-negative, we are done
            if self.alpha[min_ind] >= 0:
                break
                
            data_out = self.sv[min_ind, :]
            backup.append(data_out)
            
            # Keep only indices that are NOT the minimum index
            # Note: The original logic was: pInd = np.where(self.alpha > min(self.alpha))
            # But if there are multiple negative alphas, we should remove them one by one or carefully.
            # The standard implementation removes the most negative one first.
            
            # We need to remove the row/col from inv_A corresponding to min_ind
            # First permute it to the end
            self.inv_A = self.perm(self.inv_A, min_ind)
            # Then downdate
            self.inv_A = self.down_inv(self.inv_A)
            
            # Remove from SV and Alpha
            # Create a mask for all indices except min_ind
            mask = np.arange(len(self.alpha)) != min_ind
            self.sv = self.sv[mask]
            
            # Recompute alpha from new inv_A
            self.alpha = np.sum(self.inv_A, axis=1)
            
            if len(self.alpha) == 0:
                break

        return backup

    def perm(self, A: np.ndarray, ind: int) -> np.ndarray:
        """
        Permute the matrix A so that the row/col at `ind` moves to the last position.
        """
        n = A.shape[1]
        perm_vec = np.arange(n)
        # Shift indices: [0, ..., ind-1, ind+1, ..., n-1, ind]
        perm_vec = np.concatenate((np.arange(0, ind), np.arange(ind+1, n), [ind]))
        
        temp = A[:, perm_vec]
        res = temp[perm_vec, :]
        return res

    def score_fcn(self, new_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the anomaly score for a new data point.

        Args:
            new_data (np.ndarray): The new data point (1, n_features).

        Returns:
            Tuple[float, np.ndarray]: (Anomaly score, Similarity vector)
        """
        dist_sq = distance.cdist(new_data, self.sv)[0]
        cur_sim_vec = np.exp(-np.square(dist_sq) / (2.0 * self.sigma * self.sigma))
        m = np.max(cur_sim_vec)
        
        if m < self.eps_ol or m > 1 - self.eps_cp:
            res = -1.0
        else:
            res = self.score - np.dot(self.alpha, cur_sim_vec)
        return float(res), cur_sim_vec

    def model_update(self):
        """
        Update score and alpha values of the model after structural changes.
        """
        total_alpha = np.sum(self.alpha)
        if abs(total_alpha) < 1e-12:
             # Avoid division by zero, though this state is degenerate
             self.score = 0.0
             # self.alpha remains as is or reset? 
             # Usually implies model collapse or empty set.
        else:
            self.score = 1.0 / total_alpha
            self.alpha = self.alpha / total_alpha


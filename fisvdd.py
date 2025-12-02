import numpy as np
from scipy.spatial import distance
import time
import pandas as pd
from typing import Tuple, List, Optional

class fisvdd:
    """
    Fast Incremental Support Vector Data Description (FISVDD).
    """
    def __init__(self, data: np.ndarray, sigma: float, eps_cp: float = 1e-8, eps_ol: float = 1e-8, 
                 initial_batch_only: bool = False):
        """
        Initialize the FISVDD model.

        Args:
            data (np.ndarray): Initial dataset or initial batch to initialize the model.
            sigma (float): Gaussian kernel bandwidth.
            eps_cp (float): Epsilon for close points (numerical stability).
            eps_ol (float): Epsilon for outliers (numerical stability).
            initial_batch_only (bool): If True, only use 'data' for initialization without 
                                      storing full dataset (for incremental learning).
        """
        self.sigma = sigma
        self.eps_cp = eps_cp
        self.eps_ol = eps_ol
        
        # For incremental learning, don't store the full dataset
        if initial_batch_only:
            self.data = None
        else:
            self.data = data

        # Initialize with the first data point
        self.inv_A = np.array([[1.0]])
        self.alpha = np.array([1.0])
        self.sv = np.array([data[0]])
        self.obj_val: List[float] = []
        self.score = 1.0
        self.num_processed = 1  # Track number of points processed

    def _print_res(self):
        print("\nalpha -------")
        print(self.alpha)
        print("\nsupport vector -------")
        print(self.sv)

    def find_sv(self, data: Optional[np.ndarray] = None):
        """
        Train the FISVDD model on the initial data provided in __init__ or specified data.
        Iterates through the data points and updates the support vectors.
        
        Args:
            data (Optional[np.ndarray]): Data to process. If None, uses self.data.
        """
        # Use provided data or fallback to self.data
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no stored data available. "
                               "Pass data argument or initialize with data.")
            data = self.data
        
        for new_data in data[1:]:
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
            self.num_processed += 1

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

    def update_incremental(self, new_batch: np.ndarray, verbose: bool = False) -> dict:
        """
        Update the model incrementally with a new batch of data.
        This is the key method for batch-based incremental learning.
        
        Args:
            new_batch (np.ndarray): New batch of data points to learn from (N, n_features).
            verbose (bool): If True, print progress information.
        
        Returns:
            dict: Statistics about the update (points_processed, sv_added, sv_removed, final_sv_count).
        """
        initial_sv_count = len(self.sv)
        points_processed = 0
        sv_added = 0
        sv_removed = 0
        
        for i, new_data in enumerate(new_batch):
            new_data = np.array([new_data])
            
            score, sim_vec = self.score_fcn(new_data)
            if score > 0:
                self.expand(new_data, sim_vec)
                sv_added += 1
                
                if min(self.alpha) < 0:
                    backup = self.shrink()
                    sv_removed += len(backup)
                    
                    # Try to re-add backed up points
                    for each in backup:
                        each = np.array([each])
                        score, sim_vec = self.score_fcn(each)
                        if score > 0:
                            self.expand(each, sim_vec)
                            sv_added += 1
                
                self.model_update()
            
            self.obj_val.append(self.score)
            self.num_processed += 1
            points_processed += 1
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(new_batch)} points, SVs: {len(self.sv)}")
        
        return {
            "points_processed": points_processed,
            "sv_added": sv_added,
            "sv_removed": sv_removed,
            "initial_sv_count": initial_sv_count,
            "final_sv_count": len(self.sv),
            "total_processed": self.num_processed
        }
    
    def get_state(self) -> dict:
        """
        Get the current state of the model for checkpointing.
        
        Returns:
            dict: Model state including all necessary parameters.
        """
        return {
            "inv_A": self.inv_A.copy(),
            "alpha": self.alpha.copy(),
            "sv": self.sv.copy(),
            "sigma": self.sigma,
            "eps_cp": self.eps_cp,
            "eps_ol": self.eps_ol,
            "score": self.score,
            "obj_val": self.obj_val.copy(),
            "num_processed": self.num_processed
        }
    
    def set_state(self, state: dict):
        """
        Restore model state from a checkpoint.
        
        Args:
            state (dict): Model state dictionary from get_state().
        """
        self.inv_A = state["inv_A"].copy()
        self.alpha = state["alpha"].copy()
        self.sv = state["sv"].copy()
        self.sigma = state["sigma"]
        self.eps_cp = state["eps_cp"]
        self.eps_ol = state["eps_ol"]
        self.score = state["score"]
        self.obj_val = state["obj_val"].copy()
        self.num_processed = state["num_processed"]


import torch
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op

def _modified_gram_schmidt(dim: int) -> torch.Tensor:
    """
    Generates a random orthogonal matrix of a given dimension using the
    Modified Gram-Schmidt process.
    """
    # Start with random vectors sampled from N(0,1), as mentioned in the paper[cite: 130].
    random_vectors = torch.randn(dim, dim)
    
    basis = torch.zeros_like(random_vectors)
    for i in range(dim):
        # Start with the next random vector
        v = random_vectors[i]
        # Subtract projections onto the basis vectors found so far
        for j in range(i):
            q = basis[j]
            v = v - torch.dot(q, v) * q
        
        # Normalize the result to get the next basis vector
        norm = torch.norm(v)
        if norm < 1e-10:  # Avoid division by zero
            # This case is rare if the initial vectors are random
            raise RuntimeError("Vectors are not linearly independent.")
        basis[i] = v / norm
        
    return basis

def _create_structured_orthogonal_matrix(dim) -> torch.Tensor:
    """
    Creates a k x d orthogonal matrix for hashing using the Kronecker product.
    The small orthogonal matrices are generated using the Modified Gram-Schmidt process.
    """
    # Original case from the paper
    if dim == 64:
        # Generate three small orthogonal matrices using Gram-Schmidt
        A1 = _modified_gram_schmidt(dim=4)
        A2 = _modified_gram_schmidt(dim=4)
        A3 = _modified_gram_schmidt(dim=4)
        projection_matrix = torch.kron(torch.kron(A1, A2), A3)
        del A1, A2, A3
    # Case for D=72
    elif dim == 72:
        print("Using Gram-Schmidt and 8x8 ⊗ 9x9 Kronecker product for 72x72 matrix.")
        # Factor 72 as 8 * 9 and generate orthogonal matrices
        A1 = _modified_gram_schmidt(dim=8)
        A2 = _modified_gram_schmidt(dim=9)
        projection_matrix = torch.kron(A1, A2)
        del A1, A2
    else:
        raise ValueError(
            f"No structured matrix construction defined for d={dim}. "
            "Please add a suitable factorization in _create_structured_orthogonal_matrix."
        )
        
    return projection_matrix

class elsa_approximation:
    """
    Implements the ELSA approximation process based on the paper
    "ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention".
    
    This version uses the Modified Gram-Schmidt process to generate orthogonal vectors
    as mentioned in the paper. Modified to work with PyTorch tensors.
    """
    def __init__(self, Q, K, mx_specs, orthogonal_matrix=None):
        """
        Initializes the ELSA approximation components.
        """
        self.device = Q.device
        self.dtype = Q.dtype
        self.mx_specs = mx_specs
        self.Q = Q
        self.K = K
        self.d = Q.shape[-1]
        self.k = K.shape[-1]
        self.query_hashes = None
        self.key_hashes = None
        self.key_norms = None
        
        self.MX_Q = quantize_mx_op(
            quantize_elemwise_op(Q, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format=self.mx_specs["a_elem_format"], 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        self.MX_K = quantize_mx_op(
            quantize_elemwise_op(K, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format=self.mx_specs["a_elem_format"], 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        
        # WARNING: This bias value is from the paper for d=64, k=64[cite: 150].
        # It should be re-calibrated for other dimensions.
        self.theta_bias = 0.127
        
        # FIX: Move orthogonal matrix to the same device as Q and K
        self.projection_matrix = orthogonal_matrix.to(self.device) if orthogonal_matrix is not None else None

    def compute_hashes(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes k-bit binary hashes for each vector in the input matrix.
        """
        ## input matrix is a 4d tensor (batch_size, num_heads, seq_len, d)
        ## output matrix is a 4d tensor (batch_size, num_heads, d, k)
        projected = torch.matmul(matrix, self.projection_matrix.T)
        hashes = (projected >= 0)   #>=0 is 1, <0 is 0
        return hashes

    def approximation_scores(self) -> torch.Tensor:
        """
        Performs approximate similarity computation to find candidate keys[cite: 182].
        Returns a matrix of shape (num_queries, num_keys) with approximation scores.
        """
        # Compute pairwise Hamming distances using broadcasting
        # query_hashes: (N_q, k), key_hashes: (N_k, k)
        # Expand dimensions to enable broadcasting: (N_q, 1, k) XOR (1, N_k, k)
        # Use the original q and k inputs or their light casts
        self.query_hashes = self.compute_hashes(self.MX_Q)
        self.key_hashes = self.compute_hashes(self.MX_K)
        self.key_norms = torch.norm(self.MX_K, dim=-1)
        
        s_q = self.query_hashes.to(torch.int8).mul(2).sub(1).float()   # B H Nq k
        s_k = self.key_hashes.to(torch.int8).mul(2).sub(1).float()     # B H Nk k

        ## equals hamming distance: xor and sum
        # Result is B H Nq Nk
        
        dots = torch.einsum('bhnk,bhmk->bhnm', s_q, s_k)
        hamming = 0.5 * (self.k - dots)   # equals hamming distance
        
        # Convert Hamming distances to estimated angles
        est_angles = (torch.pi / self.k) * hamming.float()   ## pi/k (hamming distance) - theta_bias
        corrected_angles = torch.clamp(est_angles - self.theta_bias, min=0)
        
        # Broadcast key_norms to match the shape (N_q, N_k)
        key_norms_broadcasted = self.key_norms.unsqueeze(-1)  # (1, N_k)
        approx_similarities = key_norms_broadcasted * torch.cos(corrected_angles)
        # approx_similarities = torch.cos(corrected_angles)
        # approx_similarities = torch.cos(est_angles)
        return approx_similarities
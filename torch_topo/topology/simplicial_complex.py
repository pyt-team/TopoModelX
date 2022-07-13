# --------------------------------------------------------
# Constructing (co)boundary operators, Hodge Laplacians, higher order
# (co)adjacency operators for Simplicial Complexes
# Normalization of these operators are also implemented.
#
# Date: Dec 2021
# --------------------------------------------------------

import sys
from functools import lru_cache
from itertools import combinations
from warnings import warn

import numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import fractional_matrix_power
from scipy.sparse import coo_matrix, csr_matrix, diags, dok_matrix, eye
from sklearn.preprocessing import normalize

try:
    from gudhi import SimplexTree
except ImportError:
    warn(
        "gudhi library is not installed."
        + " Default computing protocol will be set for 'normal'.\n"
        + " gudhi can be installed using: 'pip install gudhi'",
        stacklevel=2,
    )


class SimplicialComplex:
    """Class for construction boundary operators, Hodge Laplacians,
    higher order (co)adjacency operators from collection of
    simplices."""

    def __init__(self, simplices, maxdimension=None, mode="gudhi"):
        self.mode = mode
        self.face_set = {}

        if not isinstance(simplices, (list, tuple)):
            raise TypeError(
                f"Input simplices must be given as a list or tuple, got {type(simplices)}."
            )

        max_simplex_size = len(max(simplices, key=lambda el: len(el)))
        if maxdimension is not None:

            filtered = [i for i in simplices if len(i) <= maxdimension + 1]
        else:
            filtered = simplices

        try:
            from gudhi import SimplexTree
        except ImportError:
            warn(
                "gudhi library is not installed."
                + "normal mode will be used for computations",
                stacklevel=2,
            )
            self.mode = "normal"
        if self.mode == "normal":

            self._import_simplices(simplices=filtered)

            if maxdimension is None:
                self.maxdim = max_simplex_size - 1
            else:

                if maxdimension > max_simplex_size - 1:
                    warn(
                        f"Maximal simplex in the collection has size {max_simplex_size}."
                        + "\n maxdimension is set to {max_simplex_size-1}",
                        stacklevel=2,
                    )
                    self.maxdim = max_simplex_size - 1
                elif maxdimension < 0:
                    raise ValueError(
                        f"maxdimension should be a positive integer, got {maxdimension}."
                    )
                else:
                    self.maxdim = maxdimension
        elif self.mode == "gudhi":
            st = self.get_simplex_tree(filtered)
            self.faces_dict = SimplicialComplex.extract_simplices(st)
            self._import_simplices(simplices=filtered)
            max_simplex_size = st.dimension() + 1
            if maxdimension is None:
                self.maxdim = max_simplex_size - 1
            else:

                if maxdimension > max_simplex_size - 1:
                    warn(
                        f"Maximal simplex in the collection has size {max_simplex_size}."
                        + f" \n {maxdimension} is set to {max_simplex_size-1}",
                        stacklevel=2,
                    )
                    self.maxdim = max_simplex_size - 1
                elif maxdimension < 0:
                    raise ValueError(
                        f"maxdimension should be a positive integer, got {maxdimension}."
                    )
                else:
                    self.maxdim = maxdimension
        else:
            raise ValueError(f" Import modes must be 'normal' and 'gudhi', got {mode}")

    @staticmethod
    def _faces(simplices):
        # valid in normal mode and can be used as a static method on any face
        # TODO, do for gudhi mode as well.
        if not isinstance(simplices, (list, tuple)):
            raise TypeError(
                f"Input simplices must be given as a list or tuple, got {type(simplices)}."
            )

        faceset = set()
        for simplex in simplices:
            numnodes = len(simplex)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex, r):
                    faceset.add(tuple(sorted(face)))
        return faceset

    def _import_simplices(self, simplices=[]):
        if self.mode == "normal":
            self.simplices = tuple(
                map(lambda simplex: tuple(sorted(simplex)), simplices)
            )
            self.face_set = SimplicialComplex._faces(self.simplices)
        elif self.mode == "gudhi":
            lst = []
            for i in range(0, len(self.faces_dict)):
                lst = lst + list(self.faces_dict[i].keys())
            self.face_set = lst

    def get_simplices(self, n):
        return self.n_faces(n)

    def n_faces(self, n):
        if n >= 0 and n <= self.maxdim:
            if self.mode == "normal":
                return tuple(filter(lambda face: len(face) == n + 1, self.face_set))
            elif self.mode == "gudhi":
                return tuple(self.faces_dict[n].keys())
        else:

            raise ValueError(
                f"dimension n should be larger than zero and not greater than {self.maxdim}\n"
                f"(maximal simplices dimension), got {n}"
            )

    def get_dimension(self):
        return self.maxdim

    def dic_order_faces(self, n):
        return sorted(self.n_faces(n))

    def get_sorted_entire_face_set(self):
        simpleces_lst = list(self.face_set)
        simpleces_lst.sort(key=lambda a: len(a))
        return simpleces_lst

    def get_simplex_tree(self, simplices):  # requires gudhi
        """
        get the simplex tree from a list of simplices
        """
        st = SimplexTree()
        for s in simplices:
            st.insert(s)
        return st

    @staticmethod
    def extract_simplices(simplex_tree):
        """
        extract simplices from gudhi simples tree
        """
        faces_dict = [dict() for _ in range(simplex_tree.dimension() + 1)]
        for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
            k = len(simplex)
            faces_dict[k - 1][frozenset(simplex)] = len(faces_dict[k - 1])
        return faces_dict

    @staticmethod
    def get_edges_from_operator(operator):
        """


        Parameters
        ----------
        operator : numpy or scipy array

        Returns
        -------
        edges : list of indices where the operator is not zero

        Rational:
        -------
         Most operaters (e.g. adjacencies/(co)boundary maps) that describe
         connectivity of the simplicial complex
         can be described as a graph whose nodes are the simplices used to
         construct the operator and whose edges correspond to the entries
         in the matrix where the operator is not zero.

         This property implies that many computations on simplicial complexes
         can be reduced to graph computations.

        """
        rows, cols = np.where(np.sign(np.abs(operator)) == 1)
        edges = zip(rows.tolist(), cols.tolist())
        return edges

    # ---------- operators ---------------#

    def boundary_operator_gudhi(self, d, signed=True):
        """
        get the boundary map using gudhi
        """
        if d == 0:
            boundary = dok_matrix(
                (1, len(self.faces_dict[d].items())), dtype=np.float32
            )
            boundary[0, 0 : len(self.faces_dict[d].items())] = 1
            return boundary.tocsr()
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in self.faces_dict[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(self.faces_dict[d - 1][face])
        assert len(values) == (d + 1) * len(self.faces_dict[d])
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(len(self.faces_dict[d - 1]), len(self.faces_dict[d])),
        )
        if signed:
            return boundary
        else:
            return abs(boundary)

    def boundary_operator_normal(self, d, signed=True):
        source_simplices = self.dic_order_faces(d)
        target_simplices = self.dic_order_faces(d - 1)

        if len(target_simplices) == 0:
            S = dok_matrix((1, len(source_simplices)), dtype=np.float32)
            S[0, 0 : len(source_simplices)] = 1
        else:
            source_simplices_dict = {
                source_simplices[j]: j for j in range(len(source_simplices))
            }
            target_simplices_dict = {
                target_simplices[i]: i for i in range(len(target_simplices))
            }

            S = dok_matrix(
                (len(target_simplices), len(source_simplices)), dtype=np.float32
            )
            for source_simplex in source_simplices:
                for a in range(len(source_simplex)):
                    target_simplex = source_simplex[:a] + source_simplex[(a + 1) :]
                    i = target_simplices_dict[target_simplex]
                    j = source_simplices_dict[source_simplex]
                    if signed:
                        S[i, j] = -1 if a % 2 == 1 else 1
                    else:
                        S[i, j] = 1
        return S

    def get_boundary_operator(self, d, signed=True):

        if d >= 0 and d <= self.maxdim:
            if self.mode == "normal":
                return self.boundary_operator_normal(d, signed).tocsr()
            elif self.mode == "gudhi":
                return self.boundary_operator_gudhi(d, signed)
        else:
            raise ValueError(
                f"d should be larget than zero and not greater than {self.maxdim} (maximal allowed dimension for simplices), got {d}"
            )

    def get_coboundary_operator(self, d, signed=True):
        return self.get_boundary_operator(d, signed).T

    def get_hodge_laplacian(self, d, signed=True):
        if d == 0:
            B_next = self.get_boundary_operator(d + 1)
            L = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.get_boundary_operator(d + 1)
            B = self.get_boundary_operator(d)
            L = B_next @ B_next.transpose() + B.transpose() @ B
        elif d == self.maxdim:
            B = self.get_boundary_operator(d)
            L = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.maxdim} (maximal dimension simplices), got {d}"
            )
        if signed:
            return L
        else:
            return abs(L)

    def get_up_laplacian(self, d, signed=True):
        if d == 0:
            B_next = self.get_boundary_operator(d + 1)
            L_up = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.get_boundary_operator(d + 1)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension simplices-1), got {d}"
            )
        if signed:
            return L_up
        else:
            return abs(L_up)

    def get_down_laplacian(self, d, signed=True):
        if d <= self.maxdim and d > 0:
            B = self.get_boundary_operator(d)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.maxdim} (maximal dimension simplices), got {d}."
            )
        if signed:
            return L_down
        else:
            return abs(L_down)

    def get_higher_order_adj(self, d, signed=False):

        L_up = self.get_up_laplacian(d, signed)

        # fast L.setdiag(0)
        # see https://github.com/scipy/scipy/issues/11600
        (nonzero,) = L_up.diagonal().nonzero()
        L_up[nonzero, nonzero] = 0
        L_up.eliminate_zeros()

        if signed:
            return L_up
        else:
            return abs(L_up)

    def get_higher_order_coadj(self, d, signed=False):

        L_down = self.get_down_laplacian(d, signed)
        # fast L.setdiag(0)
        # see https://github.com/scipy/scipy/issues/11600
        (nonzero,) = L_down.diagonal().nonzero()
        L_down[nonzero, nonzero] = 0
        L_down.eliminate_zeros()

        if signed:
            return L_down
        else:
            return abs(L_down)

    def get_k_hop_boundary(self, d, k):
        Bd = self.get_boundary_operator(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.get_higher_order_adj(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.get_higher_order_coadj(d, signed=True)
        if d == self.maxdim:
            return Bd @ np.power(coAd, k)
        elif d == 0:
            return Bd @ np.power(Ad, k)
        else:
            return Bd @ np.power(Ad, k) + Bd @ np.power(coAd, k)

    def get_k_hop_coboundary(self, d, k):
        BTd = self.get_coboundary_operator(d, signed=True)
        if d < self.maxdim and d >= 0:
            Ad = self.get_higher_order_adj(d, signed=True)
        if d <= self.maxdim and d > 0:
            coAd = self.get_higher_order_coadj(d, signed=True)
        if d == self.maxdim:
            return np.power(coAd, k) @ BTd
        elif d == 0:
            return np.power(Ad, k) @ BTd
        else:
            return np.power(Ad, k) @ BTd + np.power(coAd, k) @ BTd

    #  -----------normalized operators------------------#

    def get_normalized_hodge_laplacian(self, d, signed=True):

        return SimplicialComplex.normalize_laplacian(
            self.get_hodge_laplacian(d=d), signed=signed
        )

    def get_normalized_down_laplacian(self, d):
        Ld = self.get_hodge_laplacian(d=d)
        Ldown = self.get_down_laplacian(d=d)
        out = SimplicialComplex.normalize_x_laplacian(Ld, Ldown)
        return out

    def get_normalized_up_laplacian(self, d):
        Ld = self.get_hodge_laplacian(d=d)
        Lup = self.get_up_laplacian(d=d)
        out = SimplicialComplex.normalize_x_laplacian(Ld, Lup)
        return out

    def get_normalized_coboundary_operator(self, d, signed=True, normalization="xu"):

        CoBd = self.get_coboundary_operator(d, signed)

        if normalization == "row":
            return normalize(CoBd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(CoBd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(CoBd)
        else:
            raise Exception("invalid normalization method entered.")

    def get_normalized_k_hop_coboundary_operator(self, d, k, normalization="xu"):

        CoBd = self.get_k_hop_coboundary(d, k)

        if normalization == "row":
            return normalize(CoBd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(CoBd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(CoBd)
        else:
            raise Exception("invalid normalization method entered.")

    def get_normalized_boundary_operator(self, d, signed=True, normalization="xu"):

        Bd = self.get_boundary_operator(d, signed)
        if normalization == "row":
            return normalize(Bd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(Bd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(Bd)
        else:
            raise Exception("invalid normalization method entered.")

    def get_normalized_k_hop_boundary_operator(self, d, k, normalization="xu"):

        Bd = self.get_k_hop_boundary(d, k)
        if normalization == "row":
            return normalize(Bd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(Bd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(Bd)
        else:
            raise Exception("invalid normalization method entered.")

    def get_normalized_higher_order_adj(self, d, signed=False, normalization="kipf"):
        """
        Args:
            d: dimenion of the higher order adjacency matrix
            signed: Boolean determines if the adj matrix is signed or not.

        return:
             D^{-0.5}* (adj(d)+Id)* D^{-0.5}.
        """
        # A_adj is an opt that maps a j-cochain to a k-cochain.
        #   shape [num_of_k_simplices num_of_j_simplices]
        A_adj = self.get_higher_order_adj(d, signed=signed)
        A_adj = A_adj + eye(A_adj.shape[0])
        if normalization == "row":
            return normalize(A_adj, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.normalize_higher_order_adj(A_adj)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(A_adj)
        else:
            raise Exception("invalid normalization method entered.")

    def get_normalized_higher_order_coadj(self, d, signed=False, normalization="xu"):
        """
        Args:
            d: dimenion of the higher order adjacency matrix
            signed: Boolean determines if the adj matrix is signed or not.

        return:
             D^{-0.5}* (co-adj(d)+Id)* D^{-0.5}.
        """
        # A_adj is an opt that maps a j-cochain to a k-cochain.
        #   shape [num_of_k_simplices num_of_j_simplices]
        A_coadj = self.get_higher_order_coadj(d, signed=signed)
        A_coadj = A_coadj + eye(A_coadj.shape[0])
        if normalization == "row":
            return normalize(A_coadj, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.normalize_higher_order_adj(A_coadj)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(A_coadj)
        else:
            raise Exception("invalid normalization method entered.")

    @staticmethod
    def normalize_laplacian(L, signed=True):

        topeigen_val = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
        out = L.copy()
        out *= 1.0 / topeigen_val
        if signed:
            return out
        else:
            return abs(out)

    @staticmethod
    def normalize_x_laplacian(L, Lx):  # used to normalize the up or the down Laplacians
        assert L.shape[0] == L.shape[1]
        topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
        out = Lx.copy()
        out *= 1.0 / topeig
        return out

    @staticmethod
    def normalize_higher_order_adj(A_opt):
        """
        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
             D^{-0.5}* (A_opt)* D^{-0.5}.
        """
        rowsum = np.array(np.abs(A_opt).sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = diags(r_inv_sqrt)
        A_opt_to = A_opt.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        return coo_matrix(A_opt_to)

    @staticmethod
    def asymmetric_kipf_normalization(A_opt, is_sparse=True):
        """
        This version works for asymmetric matrices such as
        the coboundary matrices, as well as symmetric ones
        such as higher order adjacency.

        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
            a normalized version of the operator A_opt:
                D_{i}^{-0.5}* (A_opt)* D_{j}^{-0.5}
                where Di = np.sum(A_opt, axis=1)
                and Dj = np.sum(A_opt, axis=0)
        """
        if is_sparse:
            rowsum = np.array(np.abs(A_opt).sum(1))
            colsum = np.array(np.abs(A_opt).sum(0))
            degree_mat_inv_sqrt_row = diags(np.power(rowsum, -0.5).flatten())
            degree_mat_inv_sqrt_col = diags(np.power(colsum, -0.5).flatten())
            degree_mat_inv_sqrt_row = degree_mat_inv_sqrt_row.toarray()
            degree_mat_inv_sqrt_col = degree_mat_inv_sqrt_col.toarray()
            degree_mat_inv_sqrt_row[np.isinf(degree_mat_inv_sqrt_row)] = 0.0
            degree_mat_inv_sqrt_col[np.isinf(degree_mat_inv_sqrt_col)] = 0.0
            degree_mat_inv_sqrt_row = coo_matrix(degree_mat_inv_sqrt_row)
            degree_mat_inv_sqrt_col = coo_matrix(degree_mat_inv_sqrt_col)

            normalized_operator = (
                A_opt.dot(degree_mat_inv_sqrt_col)
                .transpose()
                .dot(degree_mat_inv_sqrt_row)
            ).T.tocoo()
            return normalized_operator

        else:
            Di = np.sum(np.abs(A_opt), axis=1)
            Dj = np.sum(np.abs(A_opt), axis=0)
            inv_Dj = np.array(np.diag(np.power(Dj, -0.5)))
            inv_Dj[np.isinf(inv_Dj)] = 0.0
            Di2 = np.array(np.diag(np.power(Di, -0.5)))
            Di2[np.isinf(Di2)] = 0.0
            A_opt = np.array(A_opt)
            G = Di2 @ A_opt @ inv_Dj
            return G

    @staticmethod
    def asymmetric_xu_normalization(A_opt, is_sparse=True):
        """
        This version works for asymmetric matrices such as
        the coboundary matrices, as well as symmetric ones
        such as higher order adjacency.

        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
            a normalized version of the operator A_opt:
                D_{i}^{-1}* (A_opt)
                where Di = np.sum(A_opt, axis=1)
        """
        if is_sparse:
            rowsum = np.array(np.abs(A_opt).sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = diags(r_inv)
            normalized_operator = r_mat_inv.dot(A_opt)
            return normalized_operator

        else:
            Di = np.sum(np.abs(A_opt), axis=1)
            Di2 = np.array(np.diag(np.power(Di, -1)))
            Di2[np.isinf(Di2)] = 0.0
            A_opt = np.array(A_opt)
            normalized_operator = Di2 @ A_opt
            return normalized_operator

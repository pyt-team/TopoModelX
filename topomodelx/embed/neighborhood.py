"""Functions for computing neighborhoods of a complex."""

from toponetx.classes import CellComplex, CombinatorialComplex, SimplicialComplex


def neighborhood_from_complex(
    complex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}
):
    """Compute the neighborhood of a complex.

    This function returns the indices and matrix for the neighborhood specified by `neighborhood_type`
    and `neighborhood_dim` for the input complex `complex`.

    Parameters
    ----------
    complex : SimplicialComplex or CellComplex or CombinatorialComplex or CombinatorialComplex
        The complex to compute the neighborhood for.
    neighborhood_type : str
        The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
    neighborhood_dim : dict
        The dimensions of the neighborhood to use. If `neighborhood_type` is "adj", the dimension is
        `neighborhood_dim['r']`. If `neighborhood_type` is "coadj", the dimension is `neighborhood_dim['k']`
        and `neighborhood_dim['r']` specifies the dimension of the ambient space.

        Note:
            here neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
            which the cell embeddings are going to be computed.
            r=1 means that the embeddings will be computed for the first dimension.
            The integer 'k' is ignored and only considered
            when the input complex is a combinatorial complex.

    Returns
    -------
    ind : list
        A list of the indices for the nodes in the neighborhood.
    A : ndarray
        The matrix representing the neighborhood.

    Raises
    ------
    ValueError
        If the input `complex` is not a SimplicialComplex, CellComplex, CombinatorialComplex, or
        CombinatorialComplex.
    """
    if isinstance(complex, SimplicialComplex) or isinstance(complex, CellComplex):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(neighborhood_dim["r"], index=True)

        else:
            ind, A = complex.coadjacency_matrix(neighborhood_dim["r"], index=True)
    elif isinstance(complex, CombinatorialComplex) or isinstance(
        complex, CombinatorialComplex
    ):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(
                neighborhood_dim["r"], neighborhood_dim["k"], index=True
            )
        else:
            ind, A = complex.coadjacency_matrix(
                neighborhood_dim["k"], neighborhood_dim["r"], index=True
            )
    else:
        ValueError(
            "input complex must be SimplicialComplex,CellComplex,CombinatorialComplex, or CombinatorialComplex "
        )

    return ind, A

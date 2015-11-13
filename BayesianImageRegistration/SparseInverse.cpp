#include <RcppEigen.h>

typedef Eigen::MappedSparseMatrix< double > mappedSparseMatrix ;
typedef Eigen::Map< Eigen::VectorXd > mappedVector ;

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Eigen::VectorXd cgSparse(
    const mappedSparseMatrix A,
    const mappedVector b
) {
    Eigen::ConjugateGradient< mappedSparseMatrix, Eigen::Lower > cg( A ) ;
    return cg.solve( b ) ;
}

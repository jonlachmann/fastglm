#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <algorithm>
#include <random>

#include <Rcpp.h>
#include <RcppEigen.h>

template<typename Scalar>
struct RandomRange {
    RandomRange(const Scalar& low, const Scalar& high,
                std::default_random_engine &gen) : dis(low, high), gen(gen) {}
    const Scalar operator()() const { return dis(gen); }
    mutable std::uniform_int_distribution<> dis;
    std::default_random_engine &gen;
};

Eigen::VectorXi topQuantile(Eigen::VectorXd &vec, double quantile) {
    Eigen::VectorXi inds = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size());
    auto comparator = [&vec](int a, int b){ return vec(a) > vec(b); };
    int nth = inds.size()*quantile;
    std::nth_element(inds.data(), inds.data()+nth, inds.data()+inds.size(), comparator);
    return(inds.head(nth));
}

Eigen::MatrixXd getRows(const Eigen::Map<Eigen::MatrixXd> &mat, Eigen::VectorXi &rows) {
    Eigen::MatrixXd ret(rows.size(), mat.cols());
    for (int i = 0; i < rows.size(); i++) {
        ret.row(i) = mat.row(rows(i));
    }
    return(ret);
}

Eigen::VectorXd getInds(Eigen::VectorXd &vec, Eigen::VectorXi &inds) {
    Eigen::VectorXd ret(inds.size());
    for (int i = 0; i < inds.size(); i++) {
        ret(i) = vec(inds(i));
    }
    return(ret);
}

Eigen::VectorXd getInds(const Eigen::Map<Eigen::VectorXd> &vec, Eigen::VectorXi &inds) {
    Eigen::VectorXd ret(inds.size());
    for (int i = 0; i < inds.size(); i++) {
        ret(i) = vec(inds(i));
    }
    return(ret);
}

#endif
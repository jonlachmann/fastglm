#ifndef GLM_H
#define GLM_H

#include "glm_base.h"

using Eigen::ArrayXd;
using Eigen::FullPivHouseholderQR;
using Eigen::ColPivHouseholderQR;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::HouseholderQR;
using Eigen::JacobiSVD;
using Eigen::BDCSVD;
using Eigen::LDLT;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::SelfAdjointView;
using Eigen::TriangularView;
using Eigen::VectorXd;
using Eigen::Upper;
using Eigen::EigenBase;


class glm: public GlmBase<Eigen::VectorXd, Eigen::MatrixXd> //Eigen::SparseVector<double>
{
protected:


    
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    typedef MatrixXd::Index                                 Index;
    typedef MatrixXd::Scalar                                Scalar;
    typedef MatrixXd::RealScalar                            RealScalar;
    typedef ColPivHouseholderQR<MatrixXd>::PermutationType  Permutation;
    typedef Permutation::IndicesType                        Indices;
    
    const Map<MatrixXd> X;
    const Map<VectorXd> Y;
    const Map<VectorXd> weights;
    const Map<VectorXd> offset;
    
    Function variance_fun;
    Function mu_eta_fun;
    Function linkinv;
    Function dev_resids_fun;
    Function valideta;
    Function validmu;
    
    double tol;
    int maxit;
    int maxit_s;
    int type;
    double quant;
    bool is_big_matrix;
    int debug;
    int rank;

    Eigen::Ref<Eigen::MatrixXd> X_ref;
    Eigen::Ref<Eigen::MatrixXd> Y_ref;
    Eigen::Ref<Eigen::VectorXd> weights_ref;
    Eigen::Ref<Eigen::VectorXd> offset_ref;

    FullPivHouseholderQR<MatrixXd> FPQR;
    ColPivHouseholderQR<MatrixXd> PQR;
    BDCSVD<MatrixXd> bSVD;
    HouseholderQR<MatrixXd> QR;
    LLT<MatrixXd>  Ch;
    LDLT<MatrixXd> ChD;
    JacobiSVD<MatrixXd>  UDV;
    
    SelfAdjointEigenSolver<MatrixXd> eig;
    
    Permutation                  Pmat;
    MatrixXd                     Rinv;
    VectorXd                     effects;
    
    RealScalar threshold() const 
    {
        //return m_usePrescribedThreshold ? m_prescribedThreshold
        //: numeric_limits<double>::epsilon() * nvars; 
        return numeric_limits<double>::epsilon() * nvars; 
    }
    
    // from RcppEigen
    inline ArrayXd Dplus(const ArrayXd& d) 
    {
        ArrayXd   di(d.size());
        double  comp(d.maxCoeff() * threshold());
        for (int j = 0; j < d.size(); ++j) di[j] = (d[j] < comp) ? 0. : 1./d[j];
        rank          = (di != 0.).count();
        return di;
    }
    
    MatrixXd XtWX() const 
    {
        if (!is_big_matrix || true)
        {
            return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
            rankUpdate( (w_ref.asDiagonal() * X_ref).adjoint());
        } else
        {
            MatrixXd retmat(MatrixXd::Zero(nvars, nvars));
            VectorXd wsquare = w_ref.array().square();
            for (int j = 0; j < nvars; ++j)
            {
                for (int k = j; k < nvars; ++k)
                {
                    for (int i = 0; i < X_ref.rows(); ++i)
                    {
                        retmat(j, k) += X_ref(i,j) * X_ref(i,k) * wsquare(i);
                    }
                    retmat(k, j) = retmat(j, k);
                }
            }
            return retmat;
        }
    }

    virtual void update_mu_eta()
    {
        mu_eta = Rcpp::as<Eigen::VectorXd>(mu_eta_fun(eta));
    }
    
    virtual void update_var_mu()
    {
        var_mu = Rcpp::as<Eigen::VectorXd>(variance_fun(mu));
    }
    
    virtual void update_mu()
    {
        mu = Rcpp::as<Eigen::VectorXd>(linkinv(eta));
    }
    
    virtual void update_eta()
    {
        eta = X_ref * beta + offset_ref;
    }
    
    virtual void update_z()
    {
        z = (eta - offset).array() + (Y_ref - mu).array() / mu_eta.array();
    }
    
    virtual void update_w()
    {
        w = (weights_ref.array() * mu_eta.array().square() / var_mu.array()).array().sqrt();
    }
    
    virtual void update_dev_resids()
    {
        devold = dev;
        NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
        dev = sum(dev_resids);
    }
    
    virtual void update_dev_resids_dont_update_old()
    {
        NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
        dev = sum(dev_resids);
    }
    
    virtual void step_halve()
    {
        // take half step
        beta = 0.5 * (beta.array() + beta_prev.array());
        update_eta();
        update_mu();
    }

    virtual void cooldown(int &iterr, double &ratio) {
        if (iterr > 3) {
            beta = ratio * beta.array() + (1-ratio) * beta_prev.array();
            ratio = ratio * 0.5;
        }
    }

    virtual void subsample(bool first) {
        std::default_random_engine gen(time(0));
        Eigen::VectorXi ind = Eigen::VectorXi::NullaryExpr(X.rows()*0.1,RandomRange<int>(0,X.rows()-1, gen));

        // Subset weights, offset, Y and X each time
        X_s = getRows(X, ind);
        Y_s = getInds(Y, ind);
        weights_s = getInds(weights, ind);
        offset_s = getInds(offset, ind);

        if (first) {
            // Subset initial mu and eta
            mu = getInds(mu, ind);
            eta = getInds(eta, ind);
            //new(&mu_ref) Eigen::Ref<MatrixXd>(mu_s);
            //new(&eta_ref) Eigen::Ref<VectorXd>(eta_s);

            // Change ref to weights, offset, X and Y first time
            new(&X_ref) Eigen::Ref<MatrixXd>(X_s);
            new(&Y_ref) Eigen::Ref<MatrixXd>(Y_s);
            new(&weights_ref) Eigen::Ref<MatrixXd>(weights_s);
            new(&offset_ref) Eigen::Ref<MatrixXd>(offset_s);
        } else {
            // Reset subset of mu and eta next time
            //new(&mu_ref) Eigen::Ref<MatrixXd>(mu);
            //new(&eta_ref) Eigen::Ref<VectorXd>(eta);
        }

    }

    virtual void extract_quantile()
    {
        inds = topQuantile(w, quant);
        w_s = getInds(w, inds);
        z_s = getInds(z, inds);
        X_s = getRows(X, inds);
        new(&X_ref) Eigen::Ref<MatrixXd>(X_s);
        new(&w_ref) Eigen::Ref<VectorXd>(w_s);
        new(&z_ref) Eigen::Ref<VectorXd>(z_s);
    }
    
    virtual void run_step_halving(int &iterr)
    {
        // check for infinite deviance
        if (std::isinf(dev))
        {
            int itrr = 0;
            while(std::isinf(dev))
            {
                ++itrr;
                if (itrr > maxit_s)
                {
                    break;
                }
                
                //std::cout << "half step (infinite)!" << itrr << std::endl;
                
                step_halve();
                
                // update deviance
                update_dev_resids_dont_update_old();
            }
        }
        
        // check for boundary violations
        if (!(valideta(eta) && validmu(mu)))
        {
            int itrr = 0;
            while(!(valideta(eta) && validmu(mu)))
            {
                ++itrr;
                if (itrr > 2)
                {
                    break;
                }
                
                //std::cout << "half step (boundary)!" << itrr << std::endl;
                
                step_halve();
                
            }
            
            update_dev_resids_dont_update_old();
        }
        
        
        // check for increasing deviance
        // std::abs(deviance - deviance_prev) / (0.1 + std::abs(deviance)) < tol_irls
        if ((dev - devold) / (0.1 + std::abs(dev)) >= tol && iterr > 0)
        {

            int itrr = 0;
            
            //std::cout << "dev:" << deviance << "dev prev:" << deviance_prev << std::endl;
            
            while((dev - devold) / (0.1 + std::abs(dev)) >= -tol)
            {
                ++itrr;

                if (itrr > maxit_s)
                {
                    break;
                }
                
                //std::cout << "half step (increasing dev)!" << itrr << std::endl;
                
                step_halve();
                
                
                update_dev_resids_dont_update_old();
            }
            Rcpp::Rcout << "Iters in halving " << itrr << "\n";
        }
    }
    
    // much of solve_wls() comes directly
    // from the source code of the RcppEigen package
    virtual void solve_wls(int iter)
    {
        beta_prev = beta;

        if (type == 0)
        {
            PQR.compute(w_ref.asDiagonal() * X_ref); // decompose the model matrix
            Pmat = (PQR.colsPermutation());
            rank                               = PQR.rank();
            if (rank == nvars) 
            {	// full rank case
                beta     = PQR.solve( (z_ref.array() * w_ref.array()).matrix() );
                // m_fitted   = X * m_coef;
                //m_se       = Pmat * PQR.matrixQR().topRows(m_p).
                //triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
            } else 
            {
                Rinv = (PQR.matrixQR().topLeftCorner(rank, rank).
                                                      triangularView<Upper>().
                                                      solve(MatrixXd::Identity(rank, rank)));
                effects = PQR.householderQ().adjoint() * (z_ref.array() * w_ref.array()).matrix();
                beta.head(rank)                 = Rinv * effects.head(rank);
                beta                            = Pmat * beta;
                
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                effects.tail(X_ref.rows() - rank).setZero();
                //m_fitted                          = PQR.householderQ() * effects;
                //m_se.head(m_r)                    = Rinv.rowwise().norm();
                //m_se                              = Pmat * m_se;
            }
        } else if (type == 1)
        {
            QR.compute(w_ref.asDiagonal() * X_ref);
            beta                     = QR.solve((z_ref.array() * w_ref.array()).matrix());
            //m_fitted                   = X_ref * m_coef;
            //m_se                       = QR.matrixQR().topRows(m_p).
            //triangularView<Upper>().solve(I_p()).rowwise().norm();
        } else if (type == 2)
        {
            Ch.compute(XtWX().selfadjointView<Lower>());
            beta            = Ch.solve((w_ref.asDiagonal() * X_ref).adjoint() * (z_ref.array() * w_ref.array()).matrix());
            //m_fitted          = X * m_coef;
            //m_se              = Ch.matrixL().solve(I_p()).colwise().norm();
        } else if (type == 3)
        {
            ChD.compute(XtWX().selfadjointView<Lower>());
            Dplus(ChD.vectorD());	// to set the rank
            //FIXME: Check on the permutation in the LDLT and incorporate it in
            //the coefficients and the standard error computation.
            //	m_coef            = Ch.matrixL().adjoint().
            //	    solve(Dplus(D) * Ch.matrixL().solve(X.adjoint() * y));
            beta            = ChD.solve((w_ref.asDiagonal() * X_ref).adjoint() * (z_ref.array() * w_ref.array()).matrix());
            //m_fitted          = X * m_coef;
            //m_se              = Ch.solve(I_p()).diagonal().array().sqrt();
        } else if (type == 4)
        {
            FPQR.compute(w_ref.asDiagonal() * X_ref); // decompose the model matrix
            Pmat = (FPQR.colsPermutation());
            rank                               = FPQR.rank();
            if (rank == nvars) 
            {	// full rank case
                beta     = FPQR.solve( (z_ref.array() * w_ref.array()).matrix() );
                // m_fitted   = X * m_coef;
                //m_se       = Pmat * PQR.matrixQR().topRows(m_p).
                //triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
            } else 
            {
                Rinv = (FPQR.matrixQR().topLeftCorner(rank, rank).
                            triangularView<Upper>().
                            solve(MatrixXd::Identity(rank, rank)));
                effects = FPQR.matrixQ().adjoint() * (z_ref.array() * w_ref.array()).matrix();
                //std::cout << effects.transpose() << std::endl;
                beta.head(rank)                 = Rinv * effects.head(rank);
                beta                            = Pmat * beta;
                
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                effects.tail(X_ref.rows() - rank).setZero();
                //m_fitted                          = PQR.householderQ() * effects;
                //m_se.head(m_r)                    = Rinv.rowwise().norm();
                //m_se                              = Pmat * m_se;
            }
        } else if (type == 5)
        {
            bSVD.compute(w_ref.asDiagonal() * X_ref, ComputeThinU | ComputeThinV);
            
            rank                               = bSVD.rank();
            
            // if (rank == nvars) 
            // {	// full rank case
            //     beta                     = bSVD.solve((z.array() * w.array()).matrix());
            // } else
            // {
            //     
            // }
            
            beta                     = bSVD.solve((z_ref.array() * w_ref.array()).matrix());
            
            //FIXME: Check on the permutation in the LDLT and incorporate it in
            //the coefficients and the standard error computation.
            //	m_coef            = Ch.matrixL().adjoint().
            //	    solve(Dplus(D) * Ch.matrixL().solve(X.adjoint() * y));
            //m_fitted          = X * m_coef;
            //m_se              = Ch.solve(I_p()).diagonal().array().sqrt();
        }
    }
    
    virtual void save_se()
    {
        if (type == 0)
        {
            if (rank == nvars) 
            {	// full rank case
                se       = Pmat * PQR.matrixQR().topRows(nvars).
                    triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
                return;
            } else 
            {
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                se.head(rank)                    = Rinv.rowwise().norm();
                se                               = Pmat * se;
            }
        } else if (type == 1)
        {
            se                       = QR.matrixQR().topRows(nvars).
                triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
        } else if (type == 2)
        {
            se              = Ch.matrixL().solve(MatrixXd::Identity(nvars, nvars)).colwise().norm();
        } else if (type == 3)
        {
            se              = ChD.solve(MatrixXd::Identity(nvars, nvars)).diagonal().array().sqrt();
        } else if (type == 4)
        {
            if (rank == nvars) 
            {	// full rank case
                se       = Pmat * FPQR.matrixQR().topRows(nvars).
                triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
                return;
            } else 
            {
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                se.head(rank)                    = Rinv.rowwise().norm();
                se                               = Pmat * se;
            }
        } else if (type == 5)
        {
            Rinv = (bSVD.solve(MatrixXd::Identity(nvars, nvars)));
            se                    = Rinv.rowwise().norm();
        }
        
    }
    


public:
    glm(const Map<MatrixXd> &X_,
        const Map<VectorXd> &Y_,
        const Map<VectorXd> &weights_,
        const Map<VectorXd> &offset_,
        Function &variance_fun_,
        Function &mu_eta_fun_,
        Function &linkinv_,
        Function &dev_resids_fun_,
        Function &valideta_,
        Function &validmu_,
        double tol_ = 1e-6,
        int maxit_ = 100,
        int maxit_s_ = 5,
        int type_ = 1,
        double quant_ = 1,
        bool is_big_matrix_ = false,
        int debug_ = 0) :
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(),
                                                     tol_, maxit_),
                                                     X(X_),
                                                     Y(Y_),
                                                     weights(weights_),
                                                     offset(offset_),
                                                     variance_fun(variance_fun_),
                                                     mu_eta_fun(mu_eta_fun_),
                                                     linkinv(linkinv_),
                                                     dev_resids_fun(dev_resids_fun_),
                                                     valideta(valideta_),
                                                     validmu(validmu_),
                                                     tol(tol_),
                                                     maxit(maxit_),
                                                     maxit_s(maxit_s_),
                                                     type(type_),
                                                     quant(quant_),
                                                     is_big_matrix(is_big_matrix_),
                                                     debug(debug_),
                                                     X_ref(X),
                                                     Y_ref(Y),
                                                     weights_ref(weights),
                                                     offset_ref(offset)
                                                     {}
    
    
    // must set params to starting vals
    void init_parms(const Map<VectorXd> & start_, 
                    const Map<VectorXd> & mu_,
                    const Map<VectorXd> & eta_)
    {
        beta = start_;
        eta = eta_;
        mu = mu_;
        //eta.array() += offset.array();
        
        //update_var_mu();
        
        //update_mu_eta();
        
        //update_mu();
        
        update_dev_resids();
        
        //update_z();
        
        //update_w();
        
        rank = nvars;
    }
    
    virtual VectorXd get_beta()
    {
        if (type == 0 || type == 4)
        {
            if (rank != nvars)
            {
                //beta.head(rank)                 = Rinv * effects.head(rank);
                //beta = Pmat * beta;
            }
        }
        
        return beta;
    }
    
    virtual VectorXd get_weights()  { return weights; }
    virtual int get_rank()          { return rank; }

};




#endif // GLM_H
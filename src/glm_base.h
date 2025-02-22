#ifndef GLM_BASE_H
#define GLM_BASE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include "functions.h"

#include <chrono>
#include <random>

using namespace Rcpp;
using namespace std;

using Eigen::Map;


template<typename VecTypeX, typename MatTypeX>
class GlmBase
{
protected:
    
    const int nvars;      // dimension of beta
    const int nobs;       // number of rows
    
    VecTypeX beta;        // parameters to be optimized
    VecTypeX beta_prev;   // auxiliary parameters
    
    VecTypeX eta;
    VecTypeX eta_s;
    VecTypeX var_mu;
    VecTypeX mu_eta;
    VecTypeX mu;
    VecTypeX mu_s;
    VecTypeX z;
    VecTypeX z_s;
    VecTypeX w;
    VecTypeX w_s;
    MatTypeX vcov;
    MatTypeX X_s;
    VecTypeX Y_s;
    VecTypeX weights_s;
    VecTypeX offset_s;
    VecTypeX se;
    Eigen::VectorXi inds;
    double dev, devold, devnull;
    
    int maxit;            // max iterations
    int maxit_s;          // max step halving iterations
    double tol;           // tolerance for convergence
    double quant;         // quantile of data to use
    int debug;           // debug flag
    bool conv;

    //Eigen::Ref<Eigen::VectorXd> mu_ref;
    //Eigen::Ref<Eigen::VectorXd> eta_ref;
    Eigen::Ref<Eigen::VectorXd> w_ref;
    Eigen::Ref<Eigen::VectorXd> z_ref;

    
    virtual bool converged()
    {
        if (std::abs(dev - devold)/(0.1 + std::abs(dev)) < tol)
        {
            return true;
        } else 
        {
            return false;
        }
    }
    
    
    virtual void update_eta() {}
    
    virtual void update_var_mu() {}
    
    virtual void update_mu_eta() {}
    
    virtual void update_mu() {}
    
    virtual void update_z() {}
    
    virtual void update_w() {}
    
    virtual void step_halve() {}

    virtual void cooldown(int &iterr, double &ratio) {}

    virtual void subsample(bool first) {}

    virtual void extract_quantile() {}
    
    virtual void run_step_halving(int &iterr) {}
    
    virtual void update_dev_resids() {}
    
    virtual void update_dev_resids_dont_update_old() {}
    
    virtual void solve_wls(int iter) {}
    
    virtual void save_se() {}

    
public:
    GlmBase(int n_, int p_,
            double tol_ = 1e-6,
            int maxit_ = 100) :
    nvars(p_), nobs(n_),
    beta(p_), 
    beta_prev(p_), // allocate space but do not set values
    eta(n_),
    var_mu(n_),
    mu_eta(n_),
    mu(n_),
    z(n_),
    w(n_),
    vcov(p_, p_),
    se(p_),
    maxit(maxit_),
    tol(tol_),
    //mu_ref(mu),
    //eta_ref(eta),
    w_ref(w),
    z_ref(z)
    {}
    
    virtual ~GlmBase() {}
    
    virtual void init_parms(const Map<VecTypeX> & start_, 
                            const Map<VecTypeX> & mu_,
                            const Map<VecTypeX> & eta_) {}
    
    void update_beta()
    {
        //VecTypeX newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    int solve(int maxit)
    {

        int i;

        double ratio = 0.5;
        
        conv = false;

        //subsample(true);
        
        for(i = 0; i < maxit; ++i)
        {
            if (debug == 1) Rcout << "Iteration: " << i << "\n";
            auto time0 = std::chrono::high_resolution_clock::now();

            update_var_mu();

            auto time1 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Var mu: " << std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() << "\n";

            update_mu_eta();

            auto time2 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Mu eta: " << std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() << "\n";

            update_z();

            auto time3 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Z: " << std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() << "\n";

            update_w();

            auto time4 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "W: " << std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() << "\n";

            if (quant != 1) extract_quantile();
            solve_wls(i);

            auto time5 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "WLS: " << std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() << "\n";

            //subsample(false);

            update_eta();

            auto time6 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Eta: " << std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count() << "\n";

            update_mu();

            auto time7 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Mu: " << std::chrono::duration_cast<std::chrono::microseconds>(time7 - time6).count() << "\n";

            update_dev_resids();

            auto time8 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Dev resids: " << std::chrono::duration_cast<std::chrono::microseconds>(time8 - time7).count() << "\n";

            if (quant == 1) run_step_halving(i);
            else cooldown(i, ratio);

            auto time9 = std::chrono::high_resolution_clock::now();
            if (debug == 1) Rcout << "Step halve: " << std::chrono::duration_cast<std::chrono::microseconds>(time9 - time8).count() << "\n";

            if (std::isinf(dev) && i == 0)
            {
                stop("cannot find valid starting values: please specify some");
            }
            
            if(converged())
            {
                conv = true;
                break;
            }

            
        }
        
        save_se();
        
        return std::min(i + 1, maxit);
    }
    
    virtual VecTypeX get_beta()     { return beta; }
    virtual VecTypeX get_eta()      { return eta; }
    virtual VecTypeX get_se()       { return se; }
    virtual VecTypeX get_mu()       { return mu; }
    virtual VecTypeX get_weights()  { return w; }
    virtual VecTypeX get_w()        { return w.array().square(); }
    virtual double get_dev()        { return dev; }
    virtual int get_rank()          { return nvars; }
    virtual MatTypeX get_vcov()     { return vcov; }
    virtual bool get_converged()    { return conv; }
    
};



#endif // GLM_BASE_H

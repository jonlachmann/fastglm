#ifndef GLM_BASE_H
#define GLM_BASE_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace std;


template<typename VecTypeX, typename MatTypeX>
class GlmBase
{
protected:
    
    const int nvars;      // dimension of beta
    const int nobs;       // number of rows
    
    VecTypeX beta;        // parameters to be optimized
    VecTypeX beta_prev;   // auxiliary parameters
    
    VecTypeX eta;
    VecTypeX var_mu;
    VecTypeX mu_eta;
    VecTypeX mu;
    VecTypeX z;
    VecTypeX w;
    MatTypeX vcov;
    VecTypeX se;
    double dev, devold;
    
    int maxit;            // max iterations
    double tol;           // tolerance for convergence
    
    
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
    
    
    virtual void update_eta()
    {
        
    }
    
    virtual void update_var_mu()
    {
        
    }
    
    virtual void update_mu_eta()
    {
        
    }
    
    virtual void update_mu()
    {
        
    }
    
    virtual void update_z()
    {
        
    }
    
    virtual void update_w()
    {
        
    }
    
    virtual void update_dev_resids()
    {
        
    }
    
    virtual void solve_wls()
    {
        
    }
    
    virtual void save_se()
    {
        
    }

    
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
    tol(tol_)
    {}
    
    virtual ~GlmBase() {}
    
    virtual void init_parms() {}
    
    void update_beta()
    {
        //VecTypeX newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    int solve(int maxit)
    {
        int i;
        
        for(i = 0; i < maxit; ++i)
        {
            
            update_var_mu();
            
            update_mu_eta();
            
            update_z();
            
            update_w();
            
            solve_wls();
            
            update_eta();
            
            update_mu();
            
            update_dev_resids();
            
            if(converged())
                break;
            
        }
        
        save_se();
        
        return std::min(i + 1, maxit);
    }
    
    virtual VecTypeX get_beta() { return beta; }
    virtual VecTypeX get_se()   { return se; }
    virtual MatTypeX get_vcov() { return vcov; }
    
};



#endif // GLM_BASE_H

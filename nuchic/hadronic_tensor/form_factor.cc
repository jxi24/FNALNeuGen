/**
 * \file form_factor.cc
 * Calculate the nuclear form factors
*/

#include "form_factor.hh"

#include <iostream>

std::vector<double> FormFactor::operator() (const FormFactorMode mode, double Q2) {

    // Load nucleon mass
    const double mN = _params["mN"];
    const double mN2 = mN*mN;

    // Energy parameters
    Q2 = Q2 / 1000 / 1000; // Rescale to GeV^2 since input is in MeV^2
    const double tau = Q2/4./pow(mN,2);

    // Dipole Parameters
    const double G_D=1.0/pow(1.0+Q2/pow(_params["lambda"],2), 2);
    const double G_A=_params["gan1"]/pow(1.0+Q2/pow(_params["MA"],2), 2);

    // Electric moments
    double G_ep, G_en, G_es, G_ev;

    // Magnetic moments
    const double mu_p = _params["mu_p"];
    const double mu_n = _params["mu_n"];
    double G_mp, G_mn, G_ms, G_mv;

    // Choose the appropriate mode
    switch(mode) {
        case FormFactorMode::one:
            G_ep = G_D;
            G_en = -mu_n*Q2*G_D/(1.0+Q2/mN2)/(4*mN2);

            G_mp = mu_p*G_D;
            G_mn = mu_n*G_D;

            break;
        case FormFactorMode::two:
            G_ep = (1.0 + _params["a_ep"]*tau)
                   / (1.0 + _params["b_ep0"]*tau + _params["b_ep1"]*tau*tau
                          + _params["b_ep2"]*pow(tau,3));
            G_en = G_D*_params["a_en"]*tau/(1 + _params["b_en"]*tau);

            G_mp = mu_p*(1.0 + _params["a_mp"]*tau)
                   / (1.0 + _params["b_mp0"]*tau + _params["b_mp1"]*tau*tau
                          + _params["b_mp2"]*pow(tau,3));
            G_mn = mu_p*(1.0 + _params["a_mn"]*tau)
                   / (1.0 + _params["b_mn0"]*tau + _params["b_mn1"]*tau*tau
                          + _params["b_mn2"]*pow(tau,3));

            break;
        case FormFactorMode::three: {
                // Load parameters into vectors
                std::vector<double> aa_ep = {_params["aa_ep0"], _params["aa_ep1"],
                                             _params["aa_ep2"], _params["aa_ep3"]};
                std::vector<double> bb_ep = {_params["bb_ep0"], _params["bb_ep1"],
                                             _params["bb_ep2"], _params["bb_ep3"]};
                std::vector<double> aa_mp = {_params["aa_mp0"], _params["aa_mp1"],
                                             _params["aa_mp2"], _params["aa_mp3"]};
                std::vector<double> bb_mp = {_params["bb_mp0"], _params["bb_mp1"],
                                             _params["bb_mp2"], _params["bb_mp3"]};

                std::vector<double> aa_en = {_params["aa_en0"], _params["aa_en1"],
                                             _params["aa_en2"], _params["aa_en3"]};
                std::vector<double> bb_en = {_params["bb_en0"], _params["bb_en1"],
                                             _params["bb_en2"], _params["bb_en3"]};
                std::vector<double> aa_mn = {_params["aa_mn0"], _params["aa_mn1"],
                                             _params["aa_mn2"], _params["aa_mn3"]};
                std::vector<double> bb_mn = {_params["bb_mn0"], _params["bb_mn1"],
                                             _params["bb_mn2"], _params["bb_mn3"]};

                // Initialize numerators and denominators
                double num_ep = 0, den_ep = 1, num_mp = 0, den_mp = 1;
                double num_en = 0, den_en = 1, num_mn = 0, den_mn = 1;

                for(int i = 0; i < 4; ++i) {
                    int j = i+1;
                    num_ep += aa_ep[i]*pow(tau,i);
                    den_ep += bb_ep[i]*pow(tau,j);

                    num_mp += aa_mp[i]*pow(tau,i);
                    den_mp += bb_mp[i]*pow(tau,j);

                    num_en += aa_en[i]*pow(tau,i);
                    den_en += bb_en[i]*pow(tau,j);

                    num_mn += aa_mn[i]*pow(tau,i);
                    den_mn += bb_mn[i]*pow(tau,j);
                }

                G_ep = num_ep/den_ep;
                G_mp = mu_p*num_mp/den_mp;

                G_en = num_en/den_en;
                G_mn = mu_n*num_mn/den_mn; 
            }
            break;
        default:
            throw std::runtime_error("Invalid Form Factor mode");
    }

    std::vector<double> formFactor(4,0);
    G_es = G_ep + G_en;
    G_ev = G_ep - G_en;
    G_ms = G_mp + G_mn;
    G_mv = G_mp - G_mn;

    formFactor[0] = (G_es + tau * G_ms)/(1+tau);
    formFactor[1] = (G_ev + tau * G_mv)/(1+tau);
    formFactor[2] = (G_ms - G_es)/(1+tau);
    formFactor[3] = (G_mv - G_ev)/(1+tau);

    return formFactor;
}

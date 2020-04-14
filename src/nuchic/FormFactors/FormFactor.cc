#include "nuchic/FormFactor.hh"

using nuchic::FormFactor;

FormFactor::FormFactor(const std::unordered_map<std::string, double> &params) {
    m_lambda = params.at("lambda");
    m_gan1 = params.at("gan1");
    m_ma = params.at("MA");
    m_mup = params.at("mu_p");
    m_mun = params.at("mu_n");
}

std::array<double, 4> FormFactor::Evaluate(const double &Q2) {
    const double Ges = Gep(Q2) + Gen(Q2);
    const double Gev = Gep(Q2) - Gen(Q2);
    const double Gms = Gmp(Q2) + Gmn(Q2);
    const double Gmv = Gmp(Q2) - Gmn(Q2);
    const double tau = Tau(Q2);

    return {(Ges+tau*Gms)/(1+tau),
            (Gev+tau*Gmv)/(1+tau),
            (Gms-Ges)/(1+tau),
            (Gmv-Gev)/(1+tau)};
}

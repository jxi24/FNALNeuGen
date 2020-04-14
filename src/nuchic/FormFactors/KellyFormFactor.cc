#include "nuchic/FormFactors/KellyFormFactor.hh"

using nuchic::KellyFormFactor;

KellyFormFactor::KellyFormFactor(const std::unordered_map<std::string, double> &params)
    : FormFactor(params) {

    m_aep = params.at("a_ep");
    m_bep = {params.at("b_ep0"), params.at("b_ep1"), params.at("b_ep2")};
    m_aen = params.at("a_en");
    m_ben = params.at("b_en");
    m_amp = params.at("a_mp");
    m_bmp = {params.at("b_mp0"), params.at("b_mp1"), params.at("b_mp2")};
    m_amn = params.at("a_mn");
    m_bmn = {params.at("b_mn0"), params.at("b_mn1"), params.at("b_mn2")};
}

double KellyFormFactor::Gep(const double &Q2) {
    double tau = Tau(Q2);
    return (1.0 + m_aep*tau)
        /(1.0 + m_bep[0]*tau + m_bep[1]*tau*tau + m_bep[2]*pow(tau,3));
}

double KellyFormFactor::Gen(const double &Q2) {
    return GD(Q2)*m_aen*Tau(Q2)/(1 + m_ben*Tau(Q2));
}

double KellyFormFactor::Gmp(const double &Q2) {
    double tau = Tau(Q2);
    return Mu_p()*(1 + m_amp*tau)
        / (1.0 + m_bmp[0]*tau + m_bmp[1]*tau*tau + m_bmp[2]*pow(tau,3));
}

double KellyFormFactor::Gmn(const double &Q2) {
    double tau = Tau(Q2);
    return Mu_n()*(1 + m_amp*tau)
        / (1.0 + m_bmp[0]*tau + m_bmp[1]*tau*tau + m_bmp[2]*pow(tau,3));
}

#include "nuchic/FormFactors/BBBAFormFactor.hh"

using nuchic::BBBAFormFactor;

BBBAFormFactor::BBBAFormFactor(const std::unordered_map<std::string, double> &params)
    : FormFactor(params) {

    // Proton parameters
    m_aaep = {params.at("aa_ep0"), params.at("aa_ep1"), params.at("aa_ep2"), params.at("aa_ep3")};
    m_bbep = {params.at("bb_ep0"), params.at("bb_ep1"), params.at("bb_ep2"), params.at("bb_ep3")};
    m_aamp = {params.at("aa_mp0"), params.at("aa_mp1"), params.at("aa_mp2"), params.at("aa_mp3")};
    m_bbmp = {params.at("bb_mp0"), params.at("bb_mp1"), params.at("bb_mp2"), params.at("bb_mp3")};

    // Neutron parameters
    m_aaen = {params.at("aa_en0"), params.at("aa_en1"), params.at("aa_en2"), params.at("aa_en3")};
    m_bben = {params.at("bb_en0"), params.at("bb_en1"), params.at("bb_en2"), params.at("bb_en3")};
    m_aamn = {params.at("aa_mn0"), params.at("aa_mn1"), params.at("aa_mn2"), params.at("aa_mn3")};
    m_bbmn = {params.at("bb_mn0"), params.at("bb_mn1"), params.at("bb_mn2"), params.at("bb_mn3")};
}

double BBBAFormFactor::Ratio(const double &tau, const std::array<double, 4> &a,
                             const std::array<double, 4> &b) {
    return (a[0] + a[1]*tau + a[2]*tau*tau + a[3]*pow(tau, 3))
          /(b[0]*tau + b[1]*tau*tau + b[2]*pow(tau, 3) + b[3]*pow(tau, 4));
}

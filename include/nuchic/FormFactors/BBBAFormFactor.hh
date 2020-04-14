#ifndef BBBAFORMFACTOR_HH
#define BBBAFORMFACTOR_HH

#include "nuchic/FormFactors/FormFactor.hh"

namespace nuchic {

// Form Factor from: NPB Proc. Suppl., 159, 127 (2006)
class BBBAFormFactor : public FormFactor {
    public:
        BBBAFormFactor(const std::unordered_map<std::string, double>&);

    protected:
        inline double Gep(const double &Q2) override {
            return Ratio(Tau(Q2), m_aaep, m_bbep);
        }
        inline double Gen(const double &Q2) override {
            return Ratio(Tau(Q2), m_aaen, m_bben);
        }
        double Gmp(const double &Q2) override {
            return Ratio(Tau(Q2), m_aamp, m_bbmp);
        }
        double Gmn(const double &Q2) override {
            return Ratio(Tau(Q2), m_aamn, m_bbmn);
        }

    private:
        double Ratio(const double&, const std::array<double, 4>&,
                     const std::array<double, 4>&);
        std::array<double, 4> m_aaep{}, m_bbep{}, m_aamp{}, m_bbmp{};
        std::array<double, 4> m_aaen{}, m_bben{}, m_aamn{}, m_bbmn{};
};

}

#endif

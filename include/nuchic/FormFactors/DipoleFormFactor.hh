#ifndef DIPOLEFORMFACTOR_HH
#define DIPOLEFORMFACTOR_HH

#include "nuchic/FormFactors/FormFactor.hh"

namespace nuchic {

class DipoleFormFactor : public FormFactor {
    public:
        DipoleFormFactor(const std::unordered_map<std::string, double> &params)
            : FormFactor(params) {}

    protected:
        inline double Gep(const double &Q2) override {
            return GD(Q2);
        }
        inline double Gen(const double &Q2) override {
            return -Mu_n()*Q2*GD(Q2)/(1.0+Q2/Constant::mN2)/(4*Constant::mN2);
        }
        inline double Gmp(const double &Q2) override {
            return Mu_p()*GD(Q2);
        }
        inline double Gmn(const double &Q2) override {
            return Mu_n()*GD(Q2);
        }
};

}

#endif

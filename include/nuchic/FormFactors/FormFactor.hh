#ifndef FORMFACTOR_HH
#define FORMFACTOR_HH

#include <array>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "nuchic/Constants.hh"

namespace nuchic {

enum class FormFactorType {
    Dipole,
    Kelly,          // PRC 70, 068282 (2004)
    BBBA,           // NPB Proc. Suppl., 159, 127 (2006)
    ArringtonHill   // Phys. Lett. B, 777, pg. 8-15 (2018)
};

class FormFactor {
    public:
        FormFactor(const std::unordered_map<std::string, double>&);
        FormFactor(const FormFactor&) = default;
        FormFactor(FormFactor&&) = default;
        FormFactor &operator=(const FormFactor&) = default;
        FormFactor &operator=(FormFactor&&) = default;

        virtual ~FormFactor() = default;

        std::array<double, 4> Evaluate(const double &Q2);

    protected:
        // Functions
        inline double GD(const double &Q2) {
            return 1.0/pow(1.0+Q2/pow(m_lambda, 2), 2);
        }
        inline double GA(const double &Q2) {
            return m_gan1/pow(1.0+Q2/pow(m_ma, 2), 2);
        }
        inline double Tau(const double &Q2) {
            return Q2/4/Constant::mN2;
        }
        inline double Mu_p() { return m_mup; }
        inline double Mu_n() { return m_mun; }
        virtual double Gep(const double &Q2) = 0;
        virtual double Gen(const double &Q2) = 0;
        virtual double Gmp(const double &Q2) = 0;
        virtual double Gmn(const double &Q2) = 0;

    private:
        // Variables
        double m_lambda, m_ma, m_gan1, m_mup, m_mun;
};

}


#endif

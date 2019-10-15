#ifndef FORM_FACTOR_HH
#define FORM_FACTOR_HH

#include <cmath>
#include <map>
#include <vector>

enum class FormFactorMode {
    Dipole, 
    Kelly,         // PRC 70, 068282 (2004)
    BBBA,          // NPB Proc. Suppl., 159, 127 (2006)
    Arrington-Hill // Phys. Lett. B, 777, pg. 8-15 (2018)
};

/**
 * The <code>FormFactor</code> class implements the Nuclear Form Factor calculations.
 */

class FormFactor {
    public:
        /**
         * \brief Calculate the form factor
         *
         * Initialize the class with the parameters to be used in the calculation.
         * Required parameters are:
         *  - mN: Mass of nucleon
         *  - lambda: Dipole value for G_D
         *  - gan1: Numerator for dipole G_A
         *  - MA: Value of MA for G_A dipole
         *  - mu_p: Proton magnetic moment
         *  - mu_n: Neutron magnetic moment
         *  - other: parameters associated with the specific form factor requested
         *
         *  @param[in] params Dictionary containing the parameters to be used
         */
        FormFactor(std::map<std::string, double> params) : _params(params) {};

        /**
         * Calculate the value of the form factor given a specific mode and an energy.
         *
         * @param[in] mode: The type of form factor to use
         * @param[in] Q2: The energy to calculate the form factors at, given in MeV^2
         * @return a vector containing the form factor values
         */
        std::vector<double> operator() (FormFactorMode mode, double Q2);

    private:
        std::map<std::string, double> _params;
};

#endif

#include <cmath>
#include <iostream>

#include "nuchic/EffectiveMass.hh"
#include "nuchic/Constants.hh"
#include "spdlog/spdlog.h"

double nuchic::PandharipandePieper::GetMass(const double &mass,
                                            const double &mom,
                                            const double &rho) {
    /// Computes the effective mass following Pandharipande and Pieper, 
    /// Phys.Rev.C 45 (1992) 791-798.
    /// In Eq. (2.6), the effective mass is defined using the group velocity 
    /// and the single-nucleon optical potential, U(k, rho).
    /// This function takes the parameterization of U(k, rho) from Eq. (2.16)
    /// and Eqs (2.17-2.19), which is due to Wiringa.
    /// The actual expression computed follows from these equations and a few
    /// lines of easy algebra.
    const double ratio = rho/rho0;
    const double betaTerm = beta*ratio;
    const double lambdaTerm = (lambda[0]+lambda[1]*ratio)*nuchic::Constant::HBARC;

    const double k2pL2 = pow(mom*mom + lambdaTerm*lambdaTerm, 2);

    return mass*k2pL2/(k2pL2-2*betaTerm*lambdaTerm*lambdaTerm*mass);
}

#include <cmath>

#include "nuchic/EffectiveMass.hh"
#include "nuchic/Constants.hh"

double nuchic::PandharipandePieper::GetMass(const double &mass,
                                            const double &mom,
                                            const double &rho) {
    const double ratio = rho/rho0;
    const double betaTerm = beta*ratio;
    const double lambdaTerm = (lambda[0]+lambda[1]*ratio)*nuchic::Constant::HBARC;

    const double k2pL2 = pow(mom*mom + lambdaTerm*lambdaTerm, 2);

    return mass*k2pL2/(k2pL2-2*betaTerm*lambdaTerm*mass);
}

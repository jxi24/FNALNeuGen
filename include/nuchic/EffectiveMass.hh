#ifndef EFFECTIVE_MASS_HH
#define EFFECTIVE_MASS_HH

#include <array>

namespace nuchic {

class EffectiveMass {
    public:
        EffectiveMass() = default;
        EffectiveMass(const EffectiveMass&) = default;
        EffectiveMass(EffectiveMass&&) = default;
        virtual ~EffectiveMass() = default;

        EffectiveMass& operator=(const EffectiveMass&) = default;
        EffectiveMass& operator=(EffectiveMass&&) = default;

        virtual double GetMass(const double&, const double&, const double&) = 0;
};

class PandharipandePieper : public EffectiveMass {
    public:
        PandharipandePieper(double b, std::array<double, 2> l, const double& rho)
            : beta(std::move(b)), lambda(std::move(l)), rho0{rho} {}

        double GetMass(const double&, const double&, const double&) override;

    private:
        double beta;
        std::array<double, 2> lambda;
        double rho0; 
};

}

#endif

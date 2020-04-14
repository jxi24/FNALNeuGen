#ifndef KELLYFORMFACTOR_HH
#define KELLYFORMFACTOR_HH

#include "nuchic/FormFactors/FormFactor.hh"

// Form factor from: PRC 70, 068282 (2004)
namespace nuchic {

class KellyFormFactor : FormFactor {
    public:
        KellyFormFactor(const std::unordered_map<std::string, double>&);

    protected:
        inline double Gep(const double&) override;
        inline double Gen(const double&) override;
        inline double Gmp(const double&) override;
        inline double Gmn(const double&) override;

    private:
        double m_aep, m_aen, m_ben, m_amp, m_amn;
        std::array<double, 3> m_bep{}, m_bmp{}, m_bmn{};
};

}

#endif

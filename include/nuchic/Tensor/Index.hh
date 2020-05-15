#ifndef INDEX_HH
#define INDEX_HH

#include <array>

namespace nuchic {
namespace tensor {

template<std::size_t LABEL>
class Index {
    public:
        static constexpr std::size_t value = LABEL;

        constexpr bool operator==(std::size_t other) const { return value == other; }
        constexpr bool operator!=(std::size_t other) const { return value != other; }
        
        template<std::size_t OLABEL>
        constexpr bool operator==(Index<OLABEL>) const { return LABEL == OLABEL; }
        template<std::size_t OLABEL>
        constexpr bool operator!=(Index<OLABEL>) const { return LABEL != OLABEL; }

        constexpr bool operator==(Index) const { return true; }
        constexpr bool operator!=(Index) const { return false; }

        constexpr std::size_t operator()() const { return LABEL; }
};

}

template<std::size_t LABEL>
using TensorIndex = tensor::Index<LABEL>;

static constexpr TensorIndex< 0> alpha;
static constexpr TensorIndex< 1> beta;
static constexpr TensorIndex< 2> gamma;
static constexpr TensorIndex< 3> delta;
static constexpr TensorIndex< 4> epsilon;
static constexpr TensorIndex< 5> zeta;
static constexpr TensorIndex< 6> eta;
static constexpr TensorIndex< 7> theta;
static constexpr TensorIndex< 8> iota;
static constexpr TensorIndex< 9> kappa;
static constexpr TensorIndex<10> lambda;
static constexpr TensorIndex<11> mu;
static constexpr TensorIndex<12> nu;
static constexpr TensorIndex<13> xi;
static constexpr TensorIndex<14> omicron;
static constexpr TensorIndex<15> pi;
static constexpr TensorIndex<16> rho;
static constexpr TensorIndex<17> sigma;
static constexpr TensorIndex<18> tau;
static constexpr TensorIndex<19> upsilon;
static constexpr TensorIndex<20> phi;
static constexpr TensorIndex<21> chi;
static constexpr TensorIndex<22> psi;
static constexpr TensorIndex<23> omega;

}

#endif

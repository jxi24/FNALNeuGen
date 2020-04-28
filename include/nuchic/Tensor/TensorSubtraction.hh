#ifndef TENSORSUBTRACTION_HH
#define TENSORSUBTRACTION_HH

#include "nuchic/Tensor/TensorExpression.hh"

namespace nuchic {
namespace tensor {

template<typename LHS, typename RHS, std::size_t dims>
class TensorSubtraction : public TensorExpression<TensorSubtraction<LHS, RHS, dims>, dims> {
    public:
        using size_type = std::size_t;

        static constexpr size_type rank() noexcept { return dims; }
        constexpr size_type size() const noexcept { return _size(); }

        inline constexpr TensorSubtraction(expression<LHS> _lhs,
                                           expression<RHS> _rhs) : lhs{_lhs}, rhs{_rhs} {};

        inline constexpr expression<LHS> Lhs() const { return lhs; }
        inline constexpr expression<RHS> Rhs() const { return lhs; }

        template<typename U>
        inline constexpr U eval(size_type idx) const {
            return eval_helper<LHS, RHS, U>(idx);
        }
       
    private:
        expression<LHS> lhs;
        expression<RHS> rhs;

        template<typename Lhs, typename Rhs, typename U,
                 typename std::enable_if_t<is_expression_v<Lhs> &&
                                           is_expression_v<Rhs>, bool> =0>
        inline constexpr expression<U> eval_helper(size_type idx) const {
            return lhs.template eval<U>(idx) - rhs.template eval<U>(idx);
        }

        template<typename Lhs, typename Rhs,
                 typename std::enable_if_t<is_expression_v<Lhs> &&
                                           is_expression_v<Rhs>, bool> =0>
        constexpr size_type _size() const noexcept {
            return lhs.size(); 
        }
};

template<typename LHS, typename RHS, std::size_t dims,
         typename std::enable_if_t<is_expression_v<LHS> &&
                                   is_expression_v<RHS>, bool> =0>
inline constexpr TensorSubtraction<LHS, RHS, dims>
operator-(const TensorExpression<LHS, dims> &lhs, const TensorExpression<RHS, dims> &rhs) {
    return TensorSubtraction<LHS, RHS, dims>(lhs.self(), rhs.self());
}

}
}

#endif

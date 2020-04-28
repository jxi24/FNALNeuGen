#ifndef TENSORDIVISION_HH
#define TENSORDIVISION_HH

#include "nuchic/Tensor/TensorExpression.hh"

namespace nuchic {
namespace tensor {

template<typename LHS, typename RHS, std::size_t dims>
class TensorDivision : public TensorExpression<TensorDivision<LHS, RHS, dims>, dims> {
    public:
        using size_type = std::size_t;

        static constexpr size_type rank() noexcept { return dims; }
        constexpr size_type size() const noexcept { return _size(); }

        inline constexpr TensorDivision(expression<LHS> _lhs,
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
                                           std::is_arithmetic<Rhs>::value, bool> =0>
        inline constexpr expression<U> eval_helper(size_type idx) const {
            return lhs.template eval<U>(idx) / static_cast<U>(rhs);
        }

        template<typename Lhs, typename Rhs, typename U,
                 typename std::enable_if_t<std::is_arithmetic<Lhs>::value &&
                                           is_scalar_expression<Rhs>::value, bool> =0>
        inline constexpr expression<U> eval_helper(size_type) const {
            return  static_cast<U>(lhs) / rhs.template eval<U>(0);
        }

        template<typename Lhs, typename Rhs, typename U,
                 typename std::enable_if_t<is_scalar_expression<Lhs>::value &&
                                           is_scalar_expression<Rhs>::value, bool> =0>
        inline constexpr expression<U> eval_helper(size_type) const {
            return lhs.template eval<U>(0) / rhs.template eval<U>(0);
        }

        template<typename Lhs, typename Rhs, typename U,
                 typename std::enable_if_t<is_expression_v<Lhs> && 
                                           !is_scalar_expression<Lhs>::value &&
                                           is_scalar_expression<Rhs>::value, bool> =0>
        inline constexpr expression<U> eval_helper(size_type idx) const {
            return lhs.template eval<U>(idx) / rhs.template eval<U>(0);
        }

        template<typename Lhs, typename Rhs,
                 typename std::enable_if_t<is_expression_v<Lhs> &&
                                           std::is_arithmetic<Rhs>::value, bool> =0>
        inline constexpr size_type _size() const noexcept {
            return lhs.size(); 
        }

        template<typename Lhs, typename Rhs, typename U,
                 typename std::enable_if_t<is_expression_v<Lhs> &&
                                           !is_scalar_expression<Lhs>::value &&
                                           is_scalar_expression<Rhs>::value, bool> =0>
        inline constexpr size_type _size() const {
            return lhs.size();
        }
};

template<typename LHS, typename RHS, std::size_t dims,
         typename std::enable_if_t<is_expression_v<LHS> &&
                                   std::is_arithmetic<RHS>::value, bool> =0>
inline constexpr TensorDivision<LHS, RHS, dims>
operator/(const TensorExpression<LHS, dims> &lhs, const RHS &rhs) {
    return TensorDivision<LHS, RHS, dims>(lhs.self(), rhs);
}

// Specialization for arithmetic type being divided by scalar tensor
template<typename LHS, typename RHS,
         typename std::enable_if_t<std::is_arithmetic<LHS>::value &&
                                   is_scalar_expression<RHS>::value, bool> =0>
inline constexpr TensorDivision<LHS, RHS, 0>
operator/(const LHS &lhs, const TensorExpression<RHS, 0> &rhs) {
    return TensorDivision<LHS, RHS, 0>(lhs, rhs.self());
}

// Specialization for both being scalar type
template<typename LHS, typename RHS,
         typename std::enable_if_t<is_scalar_expression<LHS>::value &&
                                   is_scalar_expression<RHS>::value, bool> =0>
inline constexpr TensorDivision<LHS, RHS, 0>
operator/(const TensorExpression<LHS, 0> &lhs, const TensorExpression<RHS, 0> &rhs) {
    return TensorDivision<LHS, RHS, 0>(lhs.self(), rhs.self());
}

// Specialization for right hand side being scalar type
template<typename LHS, typename RHS, std::size_t dims,
         typename std::enable_if_t<is_expression_v<LHS> &&
                                   !is_scalar_expression<LHS>::value &&
                                   is_scalar_expression<RHS>::value, bool> =0>
inline constexpr TensorDivision<LHS, RHS, dims>
operator/(const TensorExpression<LHS, dims> &lhs, const TensorExpression<RHS, 0> &rhs) {
    return TensorDivision<LHS, RHS, dims>(lhs.self(), rhs.self());
}

}
}

#endif

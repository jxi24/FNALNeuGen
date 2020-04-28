#ifndef TENSOREXPRESSION_HH
#define TENSOREXPRESSION_HH

// #include "nuchic/Tensor/TensorContainer.hh"

#include <type_traits>

namespace nuchic {
namespace tensor {

template<typename Expr, std::size_t dims>
class TensorExpression {
    public:
        using size_type = std::size_t;

        constexpr TensorExpression() = default;
        inline const Expr& self() const { return *static_cast<const Expr*>(this); }
        inline Expr& self() { return *static_cast<Expr*>(this); }

        static constexpr size_type rank() noexcept { return dims; }
        static constexpr size_type size() noexcept { return Expr::size(); }
};

namespace traits {

template<class Expr>
struct is_expression_impl {
    static constexpr bool value = false;
};

template<template<typename,std::size_t> class UnaryExpr, typename Expr, std::size_t dims>
struct is_expression_impl<UnaryExpr<Expr, dims>> {
    static constexpr bool value = true;
};

template<template<typename,typename,std::size_t> class BinaryExpr, typename LHS,
         typename RHS, std::size_t dims>
struct is_expression_impl<BinaryExpr<LHS, RHS, dims>> {
    static constexpr bool value = true;
};

template<class Expr, std::enable_if_t<!std::is_arithmetic<Expr>::value, bool> =0>
struct is_scalar_expr_impl {
    static constexpr bool value = (Expr::rank() == 0);
};

template<typename T>
struct BoundType {
    using type = typename std::conditional<std::is_arithmetic<T>::value, const T, const T&>::type;
};

} // end traits namespace
} // end tensor namespace

template<class Expr>
using is_expression = tensor::traits::is_expression_impl<Expr>;

template<class Expr>
static constexpr bool is_expression_v = is_expression<Expr>::value;

template<class Expr>
static constexpr bool is_expression_t = is_expression<Expr>::type;

template<class Expr>
using is_scalar_expression = tensor::traits::is_scalar_expr_impl<Expr>;

template<typename T>
using expression = typename tensor::traits::BoundType<T>::type;

} // end nuchic namespace


#endif

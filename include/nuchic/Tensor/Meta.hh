#ifndef META_HH
#define META_HH

#include <tuple>
#include <type_traits>

namespace nuchic {
namespace tensor {
namespace detail {

template<typename ...input_t>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<input_t>()...));

// Inspired by: https://stackoverflow.com/a/48204876/9201027
template<typename T, std::size_t N, typename Tuple, std::size_t... I>
constexpr decltype(auto) tuple2array_impl(const Tuple &t, std::index_sequence<I...>) {
    return std::array<T, N>{std::get<I>(t)...};
}

template<typename T, std::size_t N, typename Tuple, std::size_t... I>
constexpr decltype(auto) ptuple2parray_impl(const Tuple &t, std::index_sequence<I...>) {
    std::array<T, N> afirst{std::get<I>(t).first...};
    std::array<T, N> asecond{std::get<I>(t).second...};
    return std::make_pair(afirst, asecond);
}

}

template<typename Head, typename... Tail>
constexpr decltype(auto) tuple2array(const std::tuple<Head, Tail...> &t) {
    using Tuple = std::tuple<Head, Tail...>;
    constexpr auto N = std::tuple_size<Tuple>::value;
    return detail::tuple2array_impl<Head, N, Tuple>(t, std::make_index_sequence<N>());
}

template<typename ...Args>
constexpr decltype(auto) ptuple2parray(const std::tuple<Args...> &t) {
    using Tuple = std::tuple<Args...>;
    constexpr auto N = std::tuple_size<Tuple>::value;
    return detail::ptuple2parray_impl<std::size_t, N, Tuple>(t, std::make_index_sequence<N>());
}

// Inspired from: https://stackoverflow.com/a/18366475/9201027 and the comments within
template<template<class> class, class...>
struct filter;

template<template<class> class Pred>
struct filter<Pred> {
    using type = std::tuple<>;
};

template<template<class> class Pred, class T, class... Ts>
struct filter<Pred, T, Ts...> {
    template<class, class> struct Cons;
    template<class Head, class... Tail>
    struct Cons<Head, std::tuple<Tail...>> {
        using type = std::tuple<Head, Tail...>;
    };

    using type = typename std::conditional<
        Pred<T>::value,
        typename Cons<T, typename filter<Pred, Ts...>::type>::type,
        typename filter<Pred, Ts...>::type>::type;
};

}
}

#endif

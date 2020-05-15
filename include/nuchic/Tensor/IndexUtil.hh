#ifndef INDEXUTIL_HH
#define INDEXUTIL_HH

#include <tuple>
#include <type_traits>

namespace nuchic {
namespace tensor {
namespace detail {

// Inspired by the answer found at: https://stackoverflow.com/a/25958302/9201027
template<typename Index, typename IndexTuple>
struct has_index_impl;

template<typename Index>
struct has_index_impl<Index, std::tuple<>> : std::false_type {};

template<typename Index, typename Index0, typename ...Indices>
struct has_index_impl<Index, std::tuple<Index0, Indices...>> : has_index_impl<Index, std::tuple<Indices...>> {};

template<typename Index, typename ...Indices>
struct has_index_impl<Index, std::tuple<Index, Indices...>> : std::true_type {};

}

template<typename index_type, typename tuple_type>
struct has_index {
    static constexpr bool value = detail::has_index_impl<std::decay_t<index_type>, std::decay_t<tuple_type>>::value;
};

namespace detail {

template<typename ...index_types>
struct valid_indices_impl;

template<>
struct valid_indices_impl<std::tuple<>> : std::true_type {};

template<typename index_type>
struct valid_indices_impl<std::tuple<index_type>> : std::true_type {};

// Ensure that at most one of each index is in the indices tuple
template<typename index_type, typename ...index_types>
struct valid_indices_impl<std::tuple<index_type, index_types...>> {
    using tail = std::tuple<index_types...>;
    static constexpr bool duplicate_index = has_index<index_type, tail>::value;
    static constexpr bool value = !duplicate_index && valid_indices_impl<tail>::value;
};

}

template<typename tuple_type>
struct valid_indices {
    static constexpr bool value = detail::valid_indices_impl<std::decay_t<tuple_type>>::value;
};

namespace detail {

template<typename ...index_types>
struct num_repeated_indices_impl;

template<typename ...right>
struct num_repeated_indices_impl {
    static constexpr std::size_t value = 0;
};

template<typename head, typename ...left, typename ...right>
struct num_repeated_indices_impl<std::tuple<head, left...>, std::tuple<right...>> {
    using tuple_left = std::tuple<left...>;
    using tuple_right = std::tuple<right...>;
    static constexpr std::size_t cur_value = has_index<head, tuple_right>::value;
    static constexpr std::size_t next_value = 
        num_repeated_indices_impl<tuple_left, tuple_right>::value;
    static constexpr std::size_t value = cur_value + next_value;
};

}

template<typename tuple_left, typename tuple_right>
struct num_repeated_indices {
    static constexpr std::size_t value = 
        detail::num_repeated_indices_impl<std::decay_t<tuple_left>,
                                          std::decay_t<tuple_right>>::value;
};

namespace detail {

// Inspired by: https://stackoverflow.com/a/30736376/9201027
template<typename index, typename tuple>
struct index_position_impl;

template<typename index, typename ...indices>
struct index_position_impl<index, std::tuple<index, indices...>> 
    : std::integral_constant<std::size_t, 0> {};

template<typename index, typename oindex, typename ...indices>
struct index_position_impl<index, std::tuple<oindex, indices...>>
    : std::integral_constant<std::size_t, 
        1 + index_position_impl<index, std::tuple<indices...>>::value> {};

template<typename index>
struct index_position_impl<index, std::tuple<>> 
    : std::integral_constant<std::size_t, 0> {};

}

template<typename index, typename tuple>
struct index_position {
    using itype = std::decay_t<index>;
    using ttype = std::decay_t<tuple>;
    static constexpr bool contains = has_index<itype, ttype>::value;
    static constexpr std::size_t value = contains ?
        detail::index_position_impl<itype, ttype>::value :
        std::tuple_size<ttype>::value;
};

namespace detail {

template<std::size_t i, std::size_t N>
struct index_pairs_impl {
    template<typename array_t, typename left, typename right>
    static constexpr void run(array_t &out, const left &lhs,
                              const right &rhs, std::size_t p) {
        using index_t = std::tuple_element_t<i-1, left>;
        using has_index_t = has_index<index_t, right>;
        using get_index_t = index_position<index_t, right>;
        using next_t = index_pairs_impl<i+1, N>;
        out[p] = std::make_pair<i-1, get_index_t::value>;
        p += has_index_t::value;
        next_t::run(out, lhs, rhs, p);
    }
};

template<std::size_t N>
struct index_pairs_impl<N, N> {
    template<typename array_t, typename left, typename right>
    static constexpr void run(array_t &out, const left&,
                              const right&, std::size_t p) {
        using index_t = std::tuple_element_t<N-1, left>;
        using get_index_t = index_position<index_t, right>;
        out[p] = std::make_pair<N-1, get_index_t::value>;
    }
};

template<std::size_t i>
struct index_pairs_impl<i, 0> {
    template<typename array_t, typename left, typename right>
    static constexpr void run(array_t&, const left&, const right&, std::size_t) {}
};

}

template<typename left, typename right>
auto index_pairs(const left &lhs, const right &rhs) {
    using pair_type = std::pair<std::size_t, std::size_t>;
    constexpr auto N = std::tuple_size<left>::value;
    constexpr auto npairs = num_repeated_indices<left, right>::value;
    auto array = std::array<pair_type, npairs>{};
    detail::index_pairs_impl<1, N>::run(array, lhs, rhs, 0);
    return array;
}

}
}

#endif

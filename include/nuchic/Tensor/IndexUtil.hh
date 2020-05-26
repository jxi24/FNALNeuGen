#ifndef INDEXUTIL_HH
#define INDEXUTIL_HH

#include "nuchic/Tensor/Meta.hh"

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
template<std::size_t i, std::size_t N, typename left, typename right>
struct index_pairs_impl {
    using index_t = std::tuple_element_t<i-1, left>;
    using has_index_t = has_index<index_t, right>;
    using get_index_t = index_position<index_t, right>;
    using next_t = index_pairs_impl<i+1,N,left,right>;
    using type = typename std::conditional<has_index_t::value, 
        detail::tuple_cat_t<std::tuple<std::pair<std::integral_constant<std::size_t, i-1>, 
            std::integral_constant<std::size_t, get_index_t::value>>>, typename next_t::type>,
        detail::tuple_cat_t<typename next_t::type>>::type;
};

template<std::size_t N, typename left, typename right>
struct index_pairs_impl<N, N, left, right> {
    using index_t = std::tuple_element_t<N-1, left>;
    using has_index_t = has_index<index_t, right>;
    using get_index_t = index_position<index_t, right>;
    using type = typename std::conditional<has_index_t::value,
        std::tuple<std::pair<std::integral_constant<std::size_t, N-1>,
        std::integral_constant<std::size_t, get_index_t::value>>>, std::tuple<>>::type;
};
}

template<typename left, typename right>
struct index_pairs {
    static constexpr auto dim_left = std::tuple_size<left>::value;
    static constexpr auto size = num_repeated_indices<left, right>::value;
    using type = typename detail::index_pairs_impl<1, dim_left, left, right>::type;
    static_assert(std::tuple_size<type>::value == size, "Invalid contraction indices");
};

namespace detail {
template<std::size_t i, std::size_t N, typename left, typename right>
struct free_index_impl {
    using index_t = std::tuple_element_t<i-1, left>;
    using has_index_t = has_index<index_t, right>;
    using get_index_t = index_position<index_t, right>;
    using next_t = free_index_impl<i+1,N,left,right>;
    using type = typename std::conditional<!has_index_t::value, 
        detail::tuple_cat_t<std::tuple<std::integral_constant<std::size_t, i-1>>, typename next_t::type>,
        detail::tuple_cat_t<typename next_t::type>>::type;
};

template<std::size_t N, typename left, typename right>
struct free_index_impl<N, N, left, right> {
    using index_t = std::tuple_element_t<N-1, left>;
    using has_index_t = has_index<index_t, right>;
    using get_index_t = index_position<index_t, right>;
    using type = typename std::conditional<!has_index_t::value,
        std::tuple<std::integral_constant<std::size_t, N-1>>,
        std::tuple<>>::type;
};
}

template<typename tleft, typename tright>
struct free_indices {
    using left = std::decay_t<tleft>;
    using right = std::decay_t<tright>;
    static constexpr auto dim_left = std::tuple_size<left>::value;
    static constexpr auto size = num_repeated_indices<left, right>::value;
    using type = typename detail::free_index_impl<1, dim_left, left, right>::type;
    static_assert(dim_left - std::tuple_size<type>::value == size, "Invalid contraction indices");
};

}
}

#endif

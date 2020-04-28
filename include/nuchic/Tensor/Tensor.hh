#ifndef TENSOR_HH
#define TENSOR_HH

#include <array>
#include <vector>

#include "fmt/format.h"

#include "nuchic/Tensor/TensorExpression.hh"
#include "nuchic/Utilities.hh"

namespace nuchic {
namespace tensor {

template <typename T, std::size_t dims>
class TensorImpl : public TensorExpression<TensorImpl<T, dims>, dims> {
    private:
        static constexpr std::size_t dim_size = 4;
  
    public:
        using data_type = T;
        using data_container = std::array<data_type, ipow(dim_size, dims)>;
        using size_type = typename data_container::size_type;
    
        TensorImpl() = default;
        constexpr TensorImpl(const data_container &data) noexcept : m_data{data} {}
        template <typename U>
        inline constexpr TensorImpl(std::initializer_list<U> list) noexcept {
            std::size_t min = std::min(size(), list.size());
            std::copy_n(list.begin(), min, m_data.data());  // copy beginning
            std::fill_n(&m_data[min], size() - min, 0);     // 0-fill end
        }
        template<typename Expr>
        inline constexpr TensorImpl(TensorExpression<Expr, dims> const &expr) {
            for(size_type i = 0; i < size(); ++i) {
                m_data[i] = expr.self().template eval<T>(i);
            }
        }
    
        constexpr TensorImpl(const TensorImpl &) = default;
        constexpr TensorImpl(TensorImpl &&) = default;
        constexpr TensorImpl &operator=(const TensorImpl &) = default;
        constexpr TensorImpl &operator=(TensorImpl &&) = default;
        ~TensorImpl() = default;
    
        static constexpr inline size_type rank() noexcept { return dims; }
        static constexpr inline size_type dimension() noexcept { return dim_size; }
        static constexpr inline size_type size() noexcept {
            return ipow(dim_size, dims);
        }

        template<typename U=T>
        inline constexpr T eval(size_type idx) const {
            return m_data[idx];
        }
    
        template <typename U, std::size_t dimsU>
        constexpr bool operator==(const TensorImpl<U, dimsU> &other) const noexcept {
            static_assert(dims == dimsU,
                          "Comparison operator requires equal dimension tensors.");
            return m_data == other.m_data;
        }

        template<typename U>
        constexpr bool operator==(const U &other) const noexcept {
            static_assert(dims == 0, "Only scalars can be compared to native type.");
            return static_cast<U>(m_data[0]) == other;
        }

        template <typename U, std::size_t dimsU>
        constexpr bool operator!=(const TensorImpl<U, dimsU> &other) const noexcept {
            return !(*this == other);
        }

        template<typename U=T>
        constexpr operator U() const noexcept {
            static_assert(dims == 0, "Only scalars can be converted to native types."); 
            return static_cast<U>(m_data[0]);
        }

        template<typename U=T>
        constexpr std::array<U, size()> ToArray() const noexcept {
            return std::array<U, size()>{m_data.data()}; 
        }

        template<typename U=T>
        constexpr std::vector<U> ToVector() const noexcept {
            return std::vector<U>{m_data.data()}; 
        }

        template<typename... Args>
        constexpr const T& operator()(const Args &...args) const {
            static_assert(sizeof...(Args) == dims, "Must specify all indices of the tensor");
            return m_data[get_flat_index(static_cast<size_type>(args)...)];
        } 

        template<typename... Args>
        constexpr T& operator()(const Args &...args) {
            static_assert(sizeof...(Args) == dims, "Must specify all indices of the tensor");
            return m_data[get_flat_index(static_cast<size_type>(args)...)];
        } 

        constexpr const T& operator[](const size_type &idx) const noexcept {
            return m_data[idx];
        }

        constexpr T& operator[](const size_type &idx) noexcept {
            return m_data[idx];
        }

    private:
        constexpr size_type get_flat_index(const size_type& idx) const {
            if(idx >= dim_size) 
                throw std::range_error(fmt::format("Max dimension is {}", dim_size));
            return idx;
        }

        template<typename... Args>
        constexpr size_type get_flat_index(const size_type& idx, const Args&... args) const {
            if(idx >= dim_size) 
                throw std::range_error(fmt::format("Max dimension is {}", dim_size));
            return ipow(dim_size, sizeof...(Args))*idx + get_flat_index(args...); 
        }

        data_container m_data{};
};
}

template <typename T, std::size_t dims>
using Tensor = nuchic::tensor::TensorImpl<T, dims>;

template<typename T, std::size_t dims>
T Reduce(const Tensor<T, dims> &tensor) {
    T result{};
    for(std::size_t i = 0; i < tensor.dimension(); ++i) {
        std::size_t idx{};
        for(std::size_t j = 0; j < tensor.rank(); ++j) {
            idx += ipow(tensor.dimension(), j)*i; 
        }
        result += tensor[idx];
    } 

    return result;
}

template<typename T>
T Reduce(const Tensor<T, 0> &tensor) {
    return static_cast<T>(tensor);
} 

}

#endif

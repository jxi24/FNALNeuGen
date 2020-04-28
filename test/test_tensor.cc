#include "catch2/catch.hpp"

#include "nuchic/Tensor/Tensor.hh"
#include "nuchic/Tensor/TensorAddition.hh"
#include "nuchic/Tensor/TensorSubtraction.hh"
#include "nuchic/Tensor/TensorMultiplication.hh"
#include "nuchic/Tensor/TensorDivision.hh"

TEST_CASE("Initialization of Tensors", "[Tensor]") {
    SECTION("Scalar") {
        nuchic::Tensor<double, 0> scalar1{1};
        nuchic::Tensor<double, 0> scalar2 = {1.0};
        nuchic::Tensor<double, 0> scalar3 = scalar1;
        double scalar4 = scalar1;

        CHECK(scalar1 == scalar2);
        CHECK(scalar2 == scalar3);
        CHECK(scalar4 == static_cast<double>(scalar1));
    }

    SECTION("Vector") {
        nuchic::Tensor<double, 1> vector1{1, 2, 3, 4};
        std::array<double, 4> tmp{1, 2, 3, 4};
        nuchic::Tensor<double, 1> vector2 = tmp;
        nuchic::Tensor<double, 1> vector3 = vector1;

        CHECK(vector1 == vector2);
        CHECK(vector2 == vector3);
    }

    SECTION("Matrix") {
        nuchic::Tensor<double, 2> matrix1{ 1,  2,  3,  4,
                                           5,  6,  7,  8,
                                           9, 10, 11, 12,
                                          13, 14, 15, 16};
        std::array<double, 16> tmp{ 1,  2,  3,  4,
                                    5,  6,  7,  8,
                                    9, 10, 11, 12,
                                   13, 14, 15, 16};

        nuchic::Tensor<double, 2> matrix2 = tmp;
        nuchic::Tensor<double, 2> matrix3 = matrix1;

        CHECK(matrix1 == matrix2);
        CHECK(matrix2 == matrix3);
    }
}

TEST_CASE("Tensor Access", "[Tensor]") {
    SECTION("Constant Access") {
        nuchic::Tensor<int, 1> vector{1, 2, 3, 4};
        nuchic::Tensor<int, 2> matrix{1, 2, 3, 4,
                                      1, 2, 3, 4,
                                      1, 2, 3, 4,
                                      1, 2, 3, 4};

        // Vector
        CHECK((vector(0) == 1 && vector(1) == 2
            && vector(2) == 3 && vector(3) == 4));
        CHECK_THROWS_AS(vector(4), std::range_error);

        // Matrix
        CHECK((matrix(0, 0) == 1 && matrix(0, 1) == 2 && matrix(0, 2) == 3 && matrix(0, 3) == 4));
        CHECK((matrix(1, 0) == 1 && matrix(1, 1) == 2 && matrix(1, 2) == 3 && matrix(1, 3) == 4));
        CHECK((matrix(2, 0) == 1 && matrix(2, 1) == 2 && matrix(2, 2) == 3 && matrix(2, 3) == 4));
        CHECK((matrix(3, 0) == 1 && matrix(3, 1) == 2 && matrix(3, 2) == 3 && matrix(3, 3) == 4));
        CHECK_THROWS_AS(matrix(4, 0), std::range_error);
        CHECK_THROWS_AS(matrix(0, 4), std::range_error);
    }

    SECTION("Non-Constant Access") {
        nuchic::Tensor<int, 1> vector;
        nuchic::Tensor<int, 2> matrix;

        for(int i = 0; i < 4; ++i) {
            vector(i) = i+1;
            CHECK(vector(i) == i+1);

            for(int j = 0; j < 4; ++j) {
                matrix(i, j) = i + 1 + 4*j;
                CHECK(matrix(i, j) == i + 1 + 4*j);
            }
        }
    }
}

TEST_CASE("Tensor Reduction", "[Tensor]") {
    SECTION("Scalar Reduction") {
        // This is just the value
        nuchic::Tensor<int, 0> scalar{10};
        CHECK(nuchic::Reduce(scalar) == 10);
    }

    SECTION("Vector Reduction") {
        // This is just the sum of the vector components
        nuchic::Tensor<int, 1> vector{1, 2, 3, 4};
        CHECK(nuchic::Reduce(vector) == 10);
    }

    SECTION("Matrix Reduction") {
        // This is the trace of the matrix
        nuchic::Tensor<int, 2> matrix{1, 2, 3, 4,
                                      1, 2, 3, 4,
                                      1, 2, 3, 4,
                                      1, 2, 3, 4};
        CHECK(nuchic::Reduce(matrix) == 10);
    }

    SECTION("Tensor Reduction") {
        // This is \Sum_i A_{iii}
        nuchic::Tensor<int, 3> tensor{};
        for(std::size_t i = 0; i < 4; ++i) {
            tensor(i, i, i) = 1;
        }
        CHECK(nuchic::Reduce(tensor) == 4);
    }
}

TEST_CASE("Addition", "[Tensor]") {
    SECTION("Scalar addition") {
        nuchic::Tensor<int, 0> scalar1{10}, scalar2{5};
        nuchic::Tensor<int, 0> result{15};
        nuchic::Tensor<int, 0> sum = scalar1 + scalar2;
        nuchic::Tensor<int, 0> sum2 = scalar1 + scalar1 + scalar2;
        nuchic::Tensor<int, 0> sum3 = scalar1 + scalar2 + scalar1;

        CHECK(sum == result);
        CHECK(sum2 == sum3);
    }

    SECTION("Vector addition") {
        nuchic::Tensor<int, 1> vec1{1, 2, 3, 4};
        nuchic::Tensor<int, 1> vec2{4, 3, 2, 1};
        nuchic::Tensor<int, 1> result{5, 5, 5, 5};
        nuchic::Tensor<int, 1> sum = vec1 + vec2;
        nuchic::Tensor<int, 1> sum2 = vec1 + vec1 + vec2;
        nuchic::Tensor<int, 1> sum3 = vec1 + vec2 + vec1;

        CHECK(sum == result);
        CHECK(sum2 == sum3);
    }

    SECTION("Matrix addition") {
        nuchic::Tensor<int, 2> mat1{1, 2, 3, 4};
        nuchic::Tensor<int, 2> mat2{4, 3, 2, 1};
        nuchic::Tensor<int, 2> result{5, 5, 5, 5};
        nuchic::Tensor<int, 2> sum = mat1 + mat2;
        nuchic::Tensor<int, 2> sum2 = mat1 + mat1 + mat2;
        nuchic::Tensor<int, 2> sum3 = mat1 + mat2 + mat1;

        CHECK(sum == result);
        CHECK(sum2 == sum3);

    }
}

TEST_CASE("Subtraction", "[Tensor]") {
    SECTION("Scalar subtraction") {
        nuchic::Tensor<int, 0> scalar1{10}, scalar2{5};
        nuchic::Tensor<int, 0> result{5};
        nuchic::Tensor<int, 0> sum = scalar1 - scalar2;
        nuchic::Tensor<int, 0> sum2 = scalar1 - scalar2 - scalar1;
        nuchic::Tensor<int, 0> sum3 = scalar1 - scalar1 - scalar2;

        CHECK(sum == result);
        CHECK(sum2 == sum3);
    }

    SECTION("Vector subtraction") {
        nuchic::Tensor<int, 1> vec1{1, 2, 3, 4};
        nuchic::Tensor<int, 1> vec2{4, 3, 2, 1};
        nuchic::Tensor<int, 1> result{-3, -1, 1, 3};
        nuchic::Tensor<int, 1> sum = vec1 - vec2;
        nuchic::Tensor<int, 1> sum2 = vec1 - vec2 - vec1;
        nuchic::Tensor<int, 1> sum3 = vec1 - vec1 - vec2;

        CHECK(sum == result);
        CHECK(sum2 == sum3);
    }

    SECTION("Matrix subtraction") {
        nuchic::Tensor<int, 2> mat1{1, 2, 3, 4};
        nuchic::Tensor<int, 2> mat2{4, 3, 2, 1};
        nuchic::Tensor<int, 2> result{-3, -1, 1, 3};
        nuchic::Tensor<int, 2> sum = mat1 - mat2;
        nuchic::Tensor<int, 2> sum2 = mat1 - mat2 - mat1;
        nuchic::Tensor<int, 2> sum3 = mat1 - mat1 - mat2;

        CHECK(sum == result);
        CHECK(sum2 == sum3);
    }
}

TEST_CASE("Multiplication", "[Tensor]") {
    SECTION("Scalar multiplication") {
        nuchic::Tensor<int, 0> scalar1{10}, result{100};
        int scalar2 = 10;
        nuchic::Tensor<int, 0> scalar3{10};
        nuchic::Tensor<int, 0> product1 = scalar1*scalar2;
        nuchic::Tensor<int, 0> product2 = scalar2*scalar1;
        nuchic::Tensor<int, 0> product3 = scalar3*scalar1;
        nuchic::Tensor<int, 0> product4 = scalar1*scalar2*scalar3;
        nuchic::Tensor<int, 0> product5 = scalar3*scalar2*scalar1;

        CHECK(product1 == result);
        CHECK(product1 == product2);
        CHECK(product1 == product3);
        CHECK(product4 == product5);
    } 

    SECTION("Vector-Scalar multiplication") {
        nuchic::Tensor<int, 1> vector{1, 1, 1, 1}, result{3, 3, 3, 3};
        int scalar = 3;
        nuchic::Tensor<int, 0> scalar2{3};
        nuchic::Tensor<int, 1> product1 = vector*scalar;
        nuchic::Tensor<int, 1> product2 = scalar*vector;
        nuchic::Tensor<int, 1> product3 = vector*scalar2;
        nuchic::Tensor<int, 1> product4 = scalar2*vector;
        nuchic::Tensor<int, 1> product5 = scalar*vector*scalar2;
        nuchic::Tensor<int, 1> product6 = scalar*scalar2*vector;

        CHECK(product1 == result);
        CHECK(product1 == product2);
        CHECK(product3 == product2);
        CHECK(product3 == product4);
        CHECK(product5 == product6);
    }

    SECTION("Matrix-Scalar multiplication") {
        nuchic::Tensor<int, 2> matrix{1, 1, 1, 1}, result{3, 3, 3, 3};
        int scalar = 3;
        nuchic::Tensor<int, 0> scalar2{3};
        nuchic::Tensor<int, 2> product1 = matrix*scalar;
        nuchic::Tensor<int, 2> product2 = scalar*matrix;
        nuchic::Tensor<int, 2> product3 = matrix*scalar2;
        nuchic::Tensor<int, 2> product4 = scalar2*matrix;
        nuchic::Tensor<int, 2> product5 = scalar*matrix*scalar2;
        nuchic::Tensor<int, 2> product6 = scalar*scalar2*matrix;

        CHECK(product1 == result);
        CHECK(product1 == product2);
        CHECK(product3 == product2);
        CHECK(product3 == product4);
        CHECK(product5 == product6);
    }
}

TEST_CASE("Division", "[Tensor]") {
    SECTION("Scalar division") {
        nuchic::Tensor<int, 0> scalar1{100}, result{10};
        int scalar2 = 10;
        nuchic::Tensor<int, 0> scalar3{10};
        nuchic::Tensor<int, 0> product1 = scalar1/scalar2;
        nuchic::Tensor<int, 0> product2 = scalar1/scalar3;

        CHECK(product1 == result);
        CHECK(product1 == product2);
    } 

    SECTION("Vector-Scalar division") {
        nuchic::Tensor<int, 1> vector{3, 3, 3, 3}, result{1, 1, 1, 1};
        int scalar = 3;
        nuchic::Tensor<int, 0> scalar2{3};
        nuchic::Tensor<int, 1> product1 = vector/scalar;
        nuchic::Tensor<int, 1> product2 = vector/scalar2;

        CHECK(product1 == result);
        CHECK(product1 == product2);
    }

    SECTION("Matrix-Scalar division") {
        nuchic::Tensor<int, 2> matrix{3, 3, 3, 3}, result{1, 1, 1, 1};
        int scalar = 3;
        nuchic::Tensor<int, 0> scalar2{3};
        nuchic::Tensor<int, 2> product1 = matrix/scalar;
        nuchic::Tensor<int, 2> product2 = matrix/scalar2;

        CHECK(product1 == result);
        CHECK(product1 == product2);
    }
}

TEST_CASE("Inner Products", "[Tensor]") {
    SECTION("Scalars with Others") {

    }

    SECTION("Vectors with Vectors") {

    }

    SECTION("Vectors with Matrices") {

    }

    SECTION("Matrices with Matrices") {

    }
}

/*
TEST_CASE("Tensor Contraction", "[Tensor]") {
    SECTION("Contraction of Vectors") {
        nuchic::Tensor<double, 1> vector1{1, 2, 3, 4};
        nuchic::Tensor<double, 0> scalar1 = {30};
        double scalar2 = 30;

        nuchic::TensorIndex i;
        CHECK(vector1(i)*vector1(i) == scalar1);
        CHECK(vector1(i)*vector1(i) == scalar2);
        CHECK(Contract(vector1, vector1, i, i) == scalar1);
    }

    SECTION("Matrix Traces") {
        nuchic::Tensor<double, 2> matrix1{ 1,  2,  3,  4,
                                           5,  6,  7,  8,
                                           9, 10, 11, 12,
                                          13, 14, 15, 16};
        nuchic::Tensor<double, 0> scalar1 = {34};
        double scalar2 = 34;

        nuchic::TensorIndex i;
        CHECK(matrix1(i,i) == scalar1);
        CHECK(matrix1(i,i) == scalar2);
        CHECK(Trace(matrix1) == scalar1);
    }

    SECTION("Matrix Multiplication") {
        nuchic::Tensor<double, 2> matrix1{ 1,  2,  3,  4,
                                           5,  6,  7,  8,
                                           9, 10, 11, 12,
                                          13, 14, 15, 16};

        nuchic::Tensor<double, 2> matrix2{16, 15, 14, 13,
                                          12, 11, 10,  9,
                                           8,  7,  6,  5,
                                           4,  3,  2,  1};

        nuchic::Tensor<double, 2> result{ 80,  70,  60,  50,
                                         240, 214, 188, 162,
                                         400, 358, 316, 274,
                                         560, 502, 444, 386};

        nuchic::Tensor<double, 0> resultTrace = {996};

        nuchic::TensorIndex i, j, k;
        CHECK(matrix1(i,j)*matrix2(j,k) == result(i,k));
        CHECK(matrix1(i,j)*matrix2(j,k) == result);
        CHECK(Contract(matrix1, matrix2, i, j, j, k) == result);
        CHECK(matrix1(i,j)*matrix2(j,i) == resultTrace);
        CHECK(Contract(matrix1, matrix2, i, j, j, i) == resultTrace);
        CHECK(Trace(Contract(matrix1, matrix2, i, j, j, k)) == resultTrace);
        CHECK(Trace(matrix1(i,j)*matrix(j,k)) == resultTrace);
    }
}
*/

#include <iostream>
#include <type_traits>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#define SOME_FRACTIONAL_VALUE 12
#define ANOTHER_FRACTIONAL_VALUE 24


// General template definitions encapsulated inside a struct

template <typename float_type, typename uint_type, int fractional>
struct FloatFixedConverter {
    static float_type fixedToFloat(uint_type val) {
        static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
        static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

        using sint_type = typename std::make_signed<uint_type>::type;
        float_type scaleFactor = static_cast<float_type>(1ULL << fractional);
        float_type result = static_cast<float_type>(static_cast<sint_type>(val)) / scaleFactor;

        return result;
    }

    static uint_type floatToFixed(float_type val) {
        static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
        static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

        uint_type intPart = static_cast<uint_type>(std::abs(val));
        float_type fracPart = std::abs(val) - intPart;

        fracPart *= static_cast<float_type>(1ULL << fractional);
        uint_type fracInt = static_cast<uint_type>(fracPart + 0.5);

        uint_type result = (intPart << fractional) | fracInt;

        if (val < 0) {
            result = ~result + 1;
        }

        float_type checkValue = fixedToFloat(result);
        if (std::abs(checkValue - val) > 0.5) {
            std::cout << "floatToFixed Error: Original = " << val << ", Converted back = " << checkValue << ", Error = " << std::abs(checkValue - val) << std::endl;
        }

        return result;
    }
};

// Specialization for the case where float_type and uint_type are both float

template <int fractional>
struct FloatFixedConverter<float, float, fractional> {
    static float fixedToFloat(float val) {
        return val;
    }

    static float floatToFixed(float val) {
        return val;
    }
};

// Usage:
// FloatFixedConverter<float_type, uint_type, fractional>::fixedToFloat(val);
// FloatFixedConverter<float_type, uint_type, fractional>::floatToFixed(val);


template <typename T>
T truncate(const T& val)
{
    return val;
}

template <>
float truncate(const float& val) {
    return val; // No truncation needed for float
}


template <>
uint32_t truncate(const uint32_t& val) {
    int32_t temp = static_cast<int32_t>(val);
    temp >>= SOME_FRACTIONAL_VALUE;
    return static_cast<uint32_t>(temp);
}

template <>
uint64_t truncate(const uint64_t& val) {
    int64_t temp = static_cast<int64_t>(val);
    temp >>= ANOTHER_FRACTIONAL_VALUE;
    return static_cast<uint64_t>(temp);
}


template<typename T>
class Share{
T s1;
T s2;
    public:
Share(T s){
this->s2 = (T) rand();
this->s1 = s - this->s2;
}

Share(T s1, T s2){
this->s1 = s1;
this->s2 = s2;
}

Share(){
this->s1 = 0;
this->s2 = 0;
}

T get_s1(){
    return this->s1;
}

T get_s2(){
    return this->s2;
}

Share operator+(const Share s) const{
    return Share(this->s1 + s.s1, this->s2 + s.s2);
}

Share operator-(const Share s) const{
    return Share(this->s1 - s.s1, this->s2 - s.s2);
}

Share operator*(const Share s) const{
    auto ls1 = this->s1 * s.s1 + this->s1 * s.s2;
    auto ls2 = this->s2 * s.s1 + this->s2 * s.s2;
    return Share(ls1, ls2);
}

Share operator*(const int s) const{
    return Share(this->s1 * s, this->s2 * s);
}

Share operator/(const int s) const{
    return Share(this->s1 / s, this->s2 / s);
}

void operator+=(const Share s){
    this->s1 += s.s1;
    this->s2 += s.s2;
}

void operator-=(const Share s){
    this->s1 -= s.s1;
    this->s2 -= s.s2;
}

void operator*= (const Share s){
*this = *this * s;
}


//needed for Eigen optimization
bool operator==(const Share<T>& other) const {
    return false; 
}

Share trunc_local() const{
    auto mask = (T) rand();
    auto s1 = this->s1 + mask;
    auto s2 = this->s2 - mask;

    return Share(truncate(s1), truncate(s2));
}

template<typename float_type, int fractional>
float_type reveal_float() const{
    auto s = s1 + s2;
    /* return fixedToFloat<float_type, T, fractional>(s); */
    return FloatFixedConverter<float_type, T, fractional>::fixedToFloat(s);
    }

};

template<typename T>
class Wrapper{
T s1;
    public:
Wrapper(T s){
this->s1 = s;
}

Wrapper(){
this->s1 = 0;
}

Wrapper get_s1(){
    return this->s1;
}



Wrapper operator+(const Wrapper s) const{
    return Wrapper(this->s1 + s.s1);
}

Wrapper operator-(const Wrapper s) const{
    return Wrapper(this->s1 - s.s1);
}

Wrapper operator*(const Wrapper s) const{
    return Wrapper(this->s1 * s.s1);
}

Wrapper operator*(const int s) const{
    return Wrapper(this->s1 * s);
}

Wrapper operator/(const int s) const{
    return Wrapper(this->s1 / s);
}

void operator/=(const int s){
    this->s1 /= s;
}

void operator+=(const Wrapper s){
    this->s1 += s.s1;
}

void operator-=(const Wrapper s){
    this->s1 -= s.s1;
}

void operator*= (const Wrapper s){
*this = *this * s;
}

bool operator==(const Wrapper<T>& other) const {
    return false; 
}

//temporary solution for max pool
bool operator > (const Wrapper<T>& other) const {
    return this->s1 > other.s1;
}

Wrapper<T> relu() const{
    return Wrapper<T>(this->s1 > 0 ? this->s1 : Wrapper(0));
}

/* Wrapper<T> drelu() const{ */
/*     return Wrapper<T>(this->s1 > 0 ? Wrapper(1) : Wrapper(0)); */
/* } */

Wrapper<T> drelu(const Wrapper<T>& other) const{
    return Wrapper<T>(this->s1 > 0 ? other : Wrapper(0));
}

//temporary solutions for softmax

Wrapper<T> operator/(const Wrapper<T>& other) const{
    return Wrapper<T>(this->s1 / other.s1);
}

Wrapper<T> exp() const{
    return Wrapper<T>(std::exp(this->s1));
}

Wrapper<T> log() const{
    return Wrapper<T>(std::log(this->s1));
}

// Recursive tree-based max computation
static Wrapper<T> treeFindMax(const Wrapper<T>* begin, const Wrapper<T>* end) {
    // Base case: If there's only one element, return it
    if (end - begin == 1) {
        return *begin;
    }

    // Split data into two halves
    const Wrapper<T>* mid = begin + (end - begin) / 2;

    // Recursively find max in each half
    Wrapper<T> leftMax = treeFindMax(begin, mid);
    Wrapper<T> rightMax = treeFindMax(mid, end);

    // Return the larger of the two max values
    return (leftMax > rightMax) ? leftMax : rightMax;
}




static Wrapper<T> findMax(const Wrapper<T>* begin, const Wrapper<T>* end) {
        Wrapper<T> max_val = *begin;
        for (const Wrapper<T>* iter = begin; iter != end; ++iter) {
            if (*iter > max_val) {
                max_val = *iter;
            }
        }
        return max_val;
    }

/* static std::ptrdiff_t argMax(const Wrapper<T>* begin, const Wrapper<T>* end) { */
/*     const Wrapper<T>* max_iter = begin; */
/*     for (const Wrapper<T>* iter = begin; iter != end; ++iter) { */
/*         if (*iter > *max_iter) { */
/*             max_iter = iter; */
/*         } */
/*     } */
/*     return std::distance(begin, max_iter); */
/* } */

static void argMax(const Wrapper<T>* begin, const Wrapper<T>* end, Wrapper<T>* output) {
    const Wrapper<T>* max_iter = begin;
    for (const Wrapper<T>* iter = begin; iter != end; ++iter) {
        if (*iter > *max_iter) {
            max_iter = iter;
        }
    }
    std::ptrdiff_t max_idx = std::distance(begin, max_iter);
    output[max_idx] = T(1);
}


T reveal() const{
    return this->s1;
}

};

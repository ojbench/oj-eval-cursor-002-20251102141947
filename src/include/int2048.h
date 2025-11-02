#pragma once
#ifndef SJTU_BIGINTEGER
#define SJTU_BIGINTEGER

// Integer 1:
// Implement a signed big integer class that only needs to support simple addition and subtraction

// Integer 2:
// Implement a signed big integer class that supports addition, subtraction, multiplication, and division, and overload related operators

// Do not use any header files other than the following
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

// Do not use "using namespace std;"

namespace sjtu {
class int2048 {
  // Internal representation: little-endian base-1e9 limbs with signed flag
  // digits_[i] stores the i-th limb, base BASE
  static constexpr unsigned int BASE = 1000000000u;
  static constexpr unsigned long long BASE64 = 1000000000ull;
  std::vector<unsigned int> digits_;
  bool negative_ = false;

  // Utility helpers (do not expose)
  void trimLeadingZeros();
  static int compareAbs(const int2048 &a, const int2048 &b);
  static int2048 addAbs(const int2048 &a, const int2048 &b);
  static int2048 subAbs(const int2048 &a, const int2048 &b); // |a| >= |b|
  static int2048 mulAbsNaive(const int2048 &a, const int2048 &b);
  static void mulByDigit(const std::vector<unsigned int> &a, unsigned int m, std::vector<unsigned int> &res);
  static unsigned int divAbsByDigit(const int2048 &a, unsigned int d, int2048 &q); // returns remainder
  static unsigned long long divAbsByUint64(const int2048 &a, unsigned long long d, int2048 &q); // returns remainder
  static void addInPlaceWithShift(std::vector<unsigned int> &acc, const std::vector<unsigned int> &addend, size_t shift);
  static int2048 divmodAbs(const int2048 &a, const int2048 &b, int2048 &remainder);
  static bool isZeroVec(const std::vector<unsigned int> &v);
  bool isZero() const { return digits_.empty(); }
public:
  // Constructors
  int2048();
  int2048(long long);
  int2048(const std::string &);
  int2048(const int2048 &);

  // The parameter types of the following functions are for reference only, you can choose to use constant references or not
  // If needed, you can add other required functions yourself
  // ===================================
  // Integer1
  // ===================================

  // Read a big integer
  void read(const std::string &);
  // Output the stored big integer, no need for newline
  void print();

  // Add a big integer
  int2048 &add(const int2048 &);
  // Return the sum of two big integers
  friend int2048 add(int2048, const int2048 &);

  // Subtract a big integer
  int2048 &minus(const int2048 &);
  // Return the difference of two big integers
  friend int2048 minus(int2048, const int2048 &);

  // ===================================
  // Integer2
  // ===================================

  int2048 operator+() const;
  int2048 operator-() const;

  int2048 &operator=(const int2048 &);

  int2048 &operator+=(const int2048 &);
  friend int2048 operator+(int2048, const int2048 &);

  int2048 &operator-=(const int2048 &);
  friend int2048 operator-(int2048, const int2048 &);

  int2048 &operator*=(const int2048 &);
  friend int2048 operator*(int2048, const int2048 &);

  int2048 &operator/=(const int2048 &);
  friend int2048 operator/(int2048, const int2048 &);

  int2048 &operator%=(const int2048 &);
  friend int2048 operator%(int2048, const int2048 &);

  friend std::istream &operator>>(std::istream &, int2048 &);
  friend std::ostream &operator<<(std::ostream &, const int2048 &);

  friend bool operator==(const int2048 &, const int2048 &);
  friend bool operator!=(const int2048 &, const int2048 &);
  friend bool operator<(const int2048 &, const int2048 &);
  friend bool operator>(const int2048 &, const int2048 &);
  friend bool operator<=(const int2048 &, const int2048 &);
  friend bool operator>=(const int2048 &, const int2048 &);
};
} // namespace sjtu

#endif

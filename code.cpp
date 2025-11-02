#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// ===== Header: int2048.h =====
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

// ===== Implementation: int2048.cpp =====
namespace sjtu {

// ===== Utility helpers =====
void int2048::trimLeadingZeros() {
  while (!digits_.empty() && digits_.back() == 0) digits_.pop_back();
  if (digits_.empty()) negative_ = false;
}

bool int2048::isZeroVec(const std::vector<unsigned int> &v) { return v.empty(); }

int int2048::compareAbs(const int2048 &a, const int2048 &b) {
  if (a.digits_.size() != b.digits_.size())
    return a.digits_.size() < b.digits_.size() ? -1 : 1;
  for (size_t i = a.digits_.size(); i-- > 0;) {
    if (a.digits_[i] != b.digits_[i])
      return a.digits_[i] < b.digits_[i] ? -1 : 1;
  }
  return 0;
}

int2048 int2048::addAbs(const int2048 &a, const int2048 &b) {
  int2048 res;
  const size_t n = a.digits_.size();
  const size_t m = b.digits_.size();
  const size_t L = n > m ? n : m;
  res.digits_.assign(L + 1, 0);
  unsigned long long carry = 0;
  for (size_t i = 0; i < L; ++i) {
    unsigned long long av = i < n ? a.digits_[i] : 0;
    unsigned long long bv = i < m ? b.digits_[i] : 0;
    unsigned long long sum = av + bv + carry;
    res.digits_[i] = static_cast<unsigned int>(sum % BASE64);
    carry = sum / BASE64;
  }
  if (carry) res.digits_[L] = static_cast<unsigned int>(carry);
  res.trimLeadingZeros();
  return res;
}

int2048 int2048::subAbs(const int2048 &a, const int2048 &b) {
  // Precondition: |a| >= |b|
  int2048 res;
  res.digits_.assign(a.digits_.size(), 0);
  long long carry = 0;
  for (size_t i = 0; i < a.digits_.size(); ++i) {
    long long av = a.digits_[i];
    long long bv = i < b.digits_.size() ? b.digits_[i] : 0;
    long long cur = av - bv + carry;
    if (cur < 0) {
      cur += static_cast<long long>(BASE64);
      carry = -1;
    } else {
      carry = 0;
    }
    res.digits_[i] = static_cast<unsigned int>(cur);
  }
  res.trimLeadingZeros();
  return res;
}

void int2048::mulByDigit(const std::vector<unsigned int> &a, unsigned int m, std::vector<unsigned int> &res) {
  if (m == 0 || a.empty()) {
    res.clear();
    return;
  }
  res.assign(a.size() + 1, 0);
  unsigned long long carry = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    unsigned long long cur = static_cast<unsigned long long>(a[i]) * m + carry;
    res[i] = static_cast<unsigned int>(cur % BASE64);
    carry = cur / BASE64;
  }
  if (carry) res[a.size()] = static_cast<unsigned int>(carry);
  while (!res.empty() && res.back() == 0) res.pop_back();
}

int2048 int2048::mulAbsNaive(const int2048 &a, const int2048 &b) {
  if (a.digits_.empty() || b.digits_.empty()) return int2048(0);
  int2048 res;
  res.digits_.assign(a.digits_.size() + b.digits_.size(), 0);
  for (size_t i = 0; i < a.digits_.size(); ++i) {
    unsigned long long carry = 0;
    unsigned long long av = a.digits_[i];
    for (size_t j = 0; j < b.digits_.size() || carry; ++j) {
      unsigned long long cur = res.digits_[i + j] + av * (j < b.digits_.size() ? b.digits_[j] : 0ull) + carry;
      res.digits_[i + j] = static_cast<unsigned int>(cur % BASE64);
      carry = cur / BASE64;
    }
  }
  res.trimLeadingZeros();
  return res;
}

void int2048::addInPlaceWithShift(std::vector<unsigned int> &acc, const std::vector<unsigned int> &addend, size_t shift) {
  if (addend.empty()) return;
  if (acc.size() < shift + addend.size() + 1) acc.resize(shift + addend.size() + 1, 0);
  unsigned long long carry = 0;
  size_t i = 0;
  for (; i < addend.size(); ++i) {
    unsigned long long cur = acc[shift + i] + addend[i] + carry;
    acc[shift + i] = static_cast<unsigned int>(cur % BASE64);
    carry = cur / BASE64;
  }
  while (carry) {
    unsigned long long cur = acc[shift + i] + carry;
    acc[shift + i] = static_cast<unsigned int>(cur % BASE64);
    carry = cur / BASE64;
    ++i;
    if (shift + i >= acc.size()) acc.push_back(0);
  }
  while (!acc.empty() && acc.back() == 0) acc.pop_back();
}

unsigned int int2048::divAbsByDigit(const int2048 &a, unsigned int d, int2048 &q) {
  q.digits_.assign(a.digits_.size(), 0);
  q.negative_ = false;
  unsigned long long rem = 0;
  for (size_t i = a.digits_.size(); i-- > 0;) {
    unsigned long long cur = a.digits_[i] + rem * BASE64;
    unsigned int qi = static_cast<unsigned int>(cur / d);
    rem = cur % d;
    q.digits_[i] = qi;
  }
  q.trimLeadingZeros();
  return static_cast<unsigned int>(rem);
}

int2048 int2048::divmodAbs(const int2048 &a, const int2048 &b, int2048 &remainder) {
  // Long division: compute q = |a| / |b|, r = |a| % |b| with 0 <= r < |b|
  int2048 zero(0);
  remainder = zero;
  if (b.digits_.empty()) return zero; // undefined, but guard
  if (a.digits_.empty()) return zero;
  if (compareAbs(a, b) < 0) {
    remainder = a;
    return zero;
  }

  int2048 divisor = b;
  int2048 dividend = a;
  size_t n = divisor.digits_.size();
  size_t m = dividend.digits_.size();
  std::vector<unsigned int> q(m - n + 1, 0);

  // Normalization not strictly required if we use per-position binary search
  remainder.digits_.clear();
  remainder.negative_ = false;
  remainder.digits_.assign(m, 0);
  for (size_t i = m; i-- > 0;) {
    // Shift remainder left by one limb and add current digit
    for (size_t k = remainder.digits_.size(); k-- > 1;) {
      remainder.digits_[k] = remainder.digits_[k - 1];
    }
    if (!remainder.digits_.empty()) remainder.digits_[0] = 0;
    if (!remainder.digits_.empty()) remainder.digits_[0] = dividend.digits_[i];
    remainder.trimLeadingZeros();

    // Determine quotient digit at position i-n (if i >= n-1)
    if (compareAbs(remainder, divisor) >= 0) {
      // Binary search qdigit in [1, BASE-1]
      unsigned int lo = 0, hi = BASE - 1, best = 0;
      std::vector<unsigned int> prod;
      while (lo <= hi) {
        unsigned int mid = lo + ((hi - lo) >> 1);
        mulByDigit(divisor.digits_, mid, prod);
        int2048 prodNum;
        prodNum.digits_ = prod;
        int cmp = compareAbs(prodNum, remainder);
        if (cmp <= 0) {
          best = mid;
          lo = mid + 1;
        } else {
          if (mid == 0) break;
          hi = mid - 1;
        }
      }
      if (best) {
        std::vector<unsigned int> prodBest;
        mulByDigit(divisor.digits_, best, prodBest);
        // remainder -= prodBest
        long long carry = 0;
        size_t L = remainder.digits_.size();
        for (size_t t = 0; t < L; ++t) {
          long long rv = remainder.digits_[t];
          long long pv = t < prodBest.size() ? prodBest[t] : 0;
          long long cur = rv - pv + carry;
          if (cur < 0) {
            cur += static_cast<long long>(BASE64);
            carry = -1;
          } else {
            carry = 0;
          }
          remainder.digits_[t] = static_cast<unsigned int>(cur);
        }
        remainder.trimLeadingZeros();
      }
      if (i + 1 >= n) q[i + 1 - n] = best;
    } else if (i + 1 >= n) {
      q[i + 1 - n] = 0;
    }
  }
  // Build quotient from q[] which is little-endian aligned starting at position 0
  int2048 quotient;
  quotient.digits_.clear();
  quotient.negative_ = false;
  // q array currently has size m-n+1 where index k corresponds to limb at position k
  // Remove leading zeros
  size_t qsz = q.size();
  while (qsz > 0 && q[qsz - 1] == 0) --qsz;
  quotient.digits_.assign(q.begin(), q.begin() + qsz);
  quotient.trimLeadingZeros();
  return quotient;
}

// ===== Constructors =====
int2048::int2048() { negative_ = false; }

int2048::int2048(long long v) {
  negative_ = false;
  digits_.clear();
  if (v < 0) {
    negative_ = true;
    // Careful with LLONG_MIN
    unsigned long long x = static_cast<unsigned long long>(-(v + 1));
    x += 1ull;
    while (x) {
      digits_.push_back(static_cast<unsigned int>(x % BASE64));
      x /= BASE64;
    }
  } else {
    unsigned long long x = static_cast<unsigned long long>(v);
    while (x) {
      digits_.push_back(static_cast<unsigned int>(x % BASE64));
      x /= BASE64;
    }
  }
  trimLeadingZeros();
}

int2048::int2048(const std::string &s) { read(s); }

int2048::int2048(const int2048 &other) = default;

// ===== Basic IO and operations =====
void int2048::read(const std::string &s) {
  digits_.clear();
  negative_ = false;
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
    negative_ = (s[i] == '-');
    ++i;
  }
  // Skip leading zeros but detect if number is zero
  while (i < s.size() && s[i] == '0') ++i;
  if (i == s.size()) {
    digits_.clear();
    negative_ = false;
    return;
  }
  // Parse digit by digit: value = value * 10 + digit
  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c < '0' || c > '9') break;
    // multiply by 10
    unsigned long long carry = 0;
    for (size_t k = 0; k < digits_.size(); ++k) {
      unsigned long long cur = digits_[k] * 10ull + carry;
      digits_[k] = static_cast<unsigned int>(cur % BASE64);
      carry = cur / BASE64;
    }
    if (carry) digits_.push_back(static_cast<unsigned int>(carry));
    // add digit
    unsigned int add = static_cast<unsigned int>(c - '0');
    size_t pos = 0;
    unsigned long long c2 = add;
    while (c2) {
      if (pos >= digits_.size()) digits_.push_back(0);
      unsigned long long cur = digits_[pos] + c2;
      digits_[pos] = static_cast<unsigned int>(cur % BASE64);
      c2 = cur / BASE64;
      ++pos;
    }
  }
  trimLeadingZeros();
}

void int2048::print() {
  std::ostream &os = std::cout;
  if (digits_.empty()) {
    os << '0';
    return;
  }
  if (negative_) os << '-';
  os << digits_.back();
  for (size_t i = digits_.size() - 1; i-- > 0;) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%09u", digits_[i]);
    os << buf;
  }
}

// Add and minus (Integer1)
int2048 &int2048::add(const int2048 &other) {
  if (negative_ == other.negative_) {
    int2048 tmp = addAbs(*this, other);
    digits_.swap(tmp.digits_);
    // sign stays same
  } else {
    int cmp = compareAbs(*this, other);
    if (cmp >= 0) {
      int2048 tmp = subAbs(*this, other);
      digits_.swap(tmp.digits_);
      // sign unchanged
    } else {
      int2048 tmp = subAbs(other, *this);
      digits_.swap(tmp.digits_);
      negative_ = other.negative_;
    }
  }
  trimLeadingZeros();
  return *this;
}

int2048 add(int2048 a, const int2048 &b) { return a.add(b); }

int2048 &int2048::minus(const int2048 &other) {
  if (negative_ != other.negative_) {
    int2048 tmp = addAbs(*this, other);
    digits_.swap(tmp.digits_);
    // sign remains this->negative_
  } else {
    int cmp = compareAbs(*this, other);
    if (cmp >= 0) {
      int2048 tmp = subAbs(*this, other);
      digits_.swap(tmp.digits_);
      // sign unchanged
    } else {
      int2048 tmp = subAbs(other, *this);
      digits_.swap(tmp.digits_);
      negative_ = !negative_;
    }
  }
  trimLeadingZeros();
  return *this;
}

int2048 minus(int2048 a, const int2048 &b) { return a.minus(b); }

// ===== Operators (Integer2) =====
int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const {
  int2048 t(*this);
  if (!t.isZero()) t.negative_ = !t.negative_;
  return t;
}

int2048 &int2048::operator=(const int2048 &rhs) = default;

int2048 &int2048::operator+=(const int2048 &rhs) { return add(rhs); }
int2048 operator+(int2048 lhs, const int2048 &rhs) { return lhs += rhs; }

int2048 &int2048::operator-=(const int2048 &rhs) { return minus(rhs); }
int2048 operator-(int2048 lhs, const int2048 &rhs) { return lhs -= rhs; }

int2048 &int2048::operator*=(const int2048 &rhs) {
  if (isZero() || rhs.isZero()) {
    digits_.clear();
    negative_ = false;
    return *this;
  }
  int2048 a = *this;
  int2048 b = rhs;
  a.negative_ = b.negative_ = false;
  // Choose algorithm based on size
  size_t n = a.digits_.size();
  size_t m = b.digits_.size();
  int2048 prod;
  if ((n + m) >= 256) {
    // FFT-based multiplication in base 1e4
    // Convert to base 1e4 vectors
    const unsigned int BASE_SMALL = 10000u;
    std::vector<double> fa, fb;
    std::vector<unsigned int> A, B;
    A.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
      unsigned int x = a.digits_[i];
      A.push_back(x % BASE_SMALL);
      x /= BASE_SMALL;
      A.push_back(x % BASE_SMALL);
      x /= BASE_SMALL;
      A.push_back(x);
    }
    B.reserve(m * 3);
    for (size_t i = 0; i < m; ++i) {
      unsigned int x = b.digits_[i];
      B.push_back(x % BASE_SMALL);
      x /= BASE_SMALL;
      B.push_back(x % BASE_SMALL);
      x /= BASE_SMALL;
      B.push_back(x);
    }
    // Remove trailing zeros in A, B
    while (!A.empty() && A.back() == 0) A.pop_back();
    while (!B.empty() && B.back() == 0) B.pop_back();
    if (A.empty() || B.empty()) {
      prod = int2048(0);
    } else {
      size_t sz = 1;
      while (sz < A.size() + B.size()) sz <<= 1;
      std::vector<std::complex<double>> ca(sz), cb(sz);
      for (size_t i = 0; i < A.size(); ++i) ca[i] = std::complex<double>(A[i], 0.0);
      for (size_t i = 0; i < B.size(); ++i) cb[i] = std::complex<double>(B[i], 0.0);
      auto fft = [](std::vector<std::complex<double>> &p, bool invert) {
        size_t n = p.size();
        // bit-reverse permutation
        for (size_t i = 1, j = 0; i < n; ++i) {
          size_t bit = n >> 1;
          for (; j & bit; bit >>= 1) j ^= bit;
          j ^= bit;
          if (i < j) std::swap(p[i], p[j]);
        }
        for (size_t len = 2; len <= n; len <<= 1) {
          double ang = 2 * M_PI / len * (invert ? -1 : 1);
          std::complex<double> wlen(std::cos(ang), std::sin(ang));
          for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
              std::complex<double> u = p[i + j];
              std::complex<double> v = p[i + j + len / 2] * w;
              p[i + j] = u + v;
              p[i + j + len / 2] = u - v;
              w *= wlen;
            }
          }
        }
        if (invert) {
          for (size_t i = 0; i < n; ++i) p[i] /= static_cast<double>(n);
        }
      };
      fft(ca, false);
      fft(cb, false);
      for (size_t i = 0; i < sz; ++i) ca[i] *= cb[i];
      fft(ca, true);
      // Collect result in base 1e4
      std::vector<unsigned long long> resSmall(sz);
      for (size_t i = 0; i < sz; ++i) resSmall[i] = static_cast<unsigned long long>(std::llround(ca[i].real()));
      // Handle carries in base 1e4
      unsigned long long carry = 0;
      for (size_t i = 0; i < resSmall.size(); ++i) {
        unsigned long long cur = resSmall[i] + carry;
        resSmall[i] = static_cast<unsigned int>(cur % BASE_SMALL);
        carry = cur / BASE_SMALL;
      }
      while (carry) {
        resSmall.push_back(static_cast<unsigned int>(carry % BASE_SMALL));
        carry /= BASE_SMALL;
      }
      while (!resSmall.empty() && resSmall.back() == 0) resSmall.pop_back();
      // Convert base1e4 back to base1e9 (group 3 digits)
      prod.digits_.clear();
      prod.negative_ = false;
      for (size_t i = 0; i < resSmall.size();) {
        unsigned long long v0 = resSmall[i++];
        unsigned long long v1 = (i < resSmall.size() ? resSmall[i++] : 0ull);
        unsigned long long v2 = (i < resSmall.size() ? resSmall[i++] : 0ull);
        unsigned long long combined = v0 + v1 * 10000ull + v2 * 100000000ull; // 1e4^2 = 1e8
        prod.digits_.push_back(static_cast<unsigned int>(combined % BASE64));
        unsigned long long up = combined / BASE64;
        if (up) prod.digits_.push_back(static_cast<unsigned int>(up));
      }
      prod.trimLeadingZeros();
    }
  } else {
    prod = mulAbsNaive(a, b);
  }
  prod.negative_ = (negative_ != rhs.negative_);
  prod.trimLeadingZeros();
  digits_.swap(prod.digits_);
  negative_ = prod.negative_;
  return *this;
}
int2048 operator*(int2048 lhs, const int2048 &rhs) { return lhs *= rhs; }

int2048 &int2048::operator/=(const int2048 &rhs) {
  int2048 rem;
  bool aNeg = negative_;
  bool bNeg = rhs.negative_;
  int2048 aa = *this; aa.negative_ = false;
  int2048 bb = rhs; bb.negative_ = false;
  int2048 q = divmodAbs(aa, bb, rem);
  if (isZero()) { digits_.clear(); negative_ = false; return *this; }
  if (bNeg == aNeg) {
    // same sign -> floor == trunc
    q.negative_ = false;
    digits_.swap(q.digits_);
    negative_ = false;
  } else {
    // different signs
    if (!rem.isZero()) {
      // q = -q - 1
      q.negative_ = true; // -q
      // subtract 1
      int2048 one(1);
      q.minus(one);
      digits_ = q.digits_;
      negative_ = q.negative_;
    } else {
      q.negative_ = true;
      digits_.swap(q.digits_);
      negative_ = true;
    }
  }
  trimLeadingZeros();
  return *this;
}
int2048 operator/(int2048 lhs, const int2048 &rhs) { return lhs /= rhs; }

int2048 &int2048::operator%=(const int2048 &rhs) {
  // r = a - floor(a/b)*b; Compute via divmodAbs and adjust sign
  int2048 rem;
  bool aNeg = negative_;
  bool bNeg = rhs.negative_;
  int2048 aa = *this; aa.negative_ = false;
  int2048 bb = rhs; bb.negative_ = false;
  int2048 q = divmodAbs(aa, bb, rem);
  if (bNeg == aNeg) {
    // r is rem, non-negative
    digits_.swap(rem.digits_);
    negative_ = false;
  } else {
    if (!rem.isZero()) {
      // r = b - rem, sign same as b (negative)
      int2048 tmp = subAbs(bb, rem);
      digits_.swap(tmp.digits_);
      negative_ = true;
    } else {
      digits_.clear();
      negative_ = false;
    }
  }
  trimLeadingZeros();
  return *this;
}
int2048 operator%(int2048 lhs, const int2048 &rhs) { return lhs %= rhs; }

std::istream &operator>>(std::istream &is, int2048 &x) {
  std::string s;
  is >> s;
  x.read(s);
  return is;
}

std::ostream &operator<<(std::ostream &os, const int2048 &x) {
  if (x.digits_.empty()) { os << '0'; return os; }
  if (x.negative_) os << '-';
  os << x.digits_.back();
  for (size_t i = x.digits_.size(); i-- > 1;) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%09u", x.digits_[i - 1]);
    os << buf;
  }
  return os;
}

bool operator==(const int2048 &a, const int2048 &b) {
  return a.negative_ == b.negative_ && a.digits_ == b.digits_;
}
bool operator!=(const int2048 &a, const int2048 &b) { return !(a == b); }
bool operator<(const int2048 &a, const int2048 &b) {
  if (a.negative_ != b.negative_) return a.negative_ && !a.isZero();
  int cmp = int2048::compareAbs(a, b);
  if (a.negative_) return cmp > 0; // both negative
  return cmp < 0;
}
bool operator>(const int2048 &a, const int2048 &b) { return b < a; }
bool operator<=(const int2048 &a, const int2048 &b) { return !(b < a); }
bool operator>=(const int2048 &a, const int2048 &b) { return !(a < b); }

} // namespace sjtu

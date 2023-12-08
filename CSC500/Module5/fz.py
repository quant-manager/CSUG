#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
This module implements Fuzzy Logic class for estimating values of fuzzy logic
expressions. Fuzzy logic replaces Boolean types with Bernoulli distribution
random variables that assign probability that the Boolean value is True.

Usage: python fz.py

Documentation style: https://realpython.com/documenting-python-code/

Revision History:

    1. 12/7/2023: Initial Draft.
    2. 12/8/2023: Updated comments regarding conditional probabilities feature,
                  which can be added.

@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""

###############################################################################
# Bayes Theorem Formula:
#
# P(A | B) = P(A and B) / P(B) = [P(B | A) * P(A)] / P(B).
# P(A and B) == P(A | B) * P(B) == P(B | A) * P(A).
# P(A and B) == P(A) * P(B) iff A and B are independent.
# P(A and B) == P(A) == P(B) iff A is B (identity),
#     but not just A == B (equality).
# P(A or B) == P(A) + P(B) iff A and B are
#     mutually exclusive (assumption not used).
# P(A or B) == P(A) + P(B) - P(A and B) iff A and B are inclusive.
#           == P(A) + P(B) - P(A | B) * P(B) == P(A) + P(B) - P(B | A) * P(A)
###############################################################################
# In this module, consider either independent A and B, or identical A and B
# ("A is B", not just "A == B"). Assume that conditional probabilities are:
# 1. P (A | B) = P(A) and P(B | A) = P(B) for independent A and B
# 2. P (A | B) = 1 and P(B | A) = 1 for identical A and B (equal A and B
#    may not be identical!)
#
# Independent A and B:
#
# P(A and B) == P(A | B) * P(B) == P(B | A) * P(A) = P(A) * P(B)
# P(A or B) == P(A) + P(B) - P(A and B) = P(A) + P(B) - P(A) * P(B)
# P(not A) == 1 - P(A)
#
# Identical A and B:
#
# P(A and B) == P(A | B) * P(B) == P(A | A) * P(A) = 1 * P(A) = P(A) = P(B)
# P(A or B) == P(A) + P(B) - P(A and B) = P(A) + P(B) - P(A) = P(A) = P(B)
# P(not A) == 1 - P(A)
#
###############################################################################
# Do not consider input of conditional probabilities P(A | B) to compute
# P(A and B) == P(A | B) * P(B) == P(B | A) * P(A).
# Otherwise, that would require providing a conditional probability to each
# fuzzy operator, regardless whether it is applied to and atomic or composite
# operands. The following example demonstrates the point:
#    x1.And(x2, P_x1_given_x2).Or(x3, P_both_x1_and_x2_given_x3).
# The Bayes Theorem Formula can be used to derive the symmetrical conditional
# probability, as needed:
# P(A | B) = [P(B | A) * P(A)] / P(B).
# P(B | A) = [P(A | B) * P(B)] / P(A).
# During fuzzy logic expression evaluation, the products P(A | B) * P(B) would
# be used in calculations instead of P(A) * P(B). This feature is not
# implemented yet!
###############################################################################


from decimal import Decimal, getcontext
from mpmath import mpf, mp
from sys import float_info


class Fzb :

    INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT = 50 # impacts DEC_EPS and MPF_EPS
    FLT_EPS = float_info.epsilon # 2.220446049250313e-16
    # Use _generate_epsilon to update epsilons, if necessary
    DEC_EPS = Decimal("8.552847072295026067649716694884742012208630016336E-50")
    MPF_EPS = mpf("2.672764710092195646140536467151481878815196880105e-51")
    getcontext().prec = INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT
    mp.dps = INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT

    # https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python
    def _generate_epsilon(self):
        eps = (self._val_type)(+1.) # float, Decimal, or mpf
        while eps + 1 > 1:
            eps /= 2
        eps *= 2
        return eps

    def __init__(self, val = None) :
        if val is not None :
            if not isinstance(val, bool) and not isinstance(val, float) and \
               not isinstance(val, Decimal) and not isinstance(val, mpf) :
                raise TypeError("Invalid value type for fuzzy boolean")
            _val = val if not isinstance(val, bool) else (float)(val)
            self._val = _val
            self._val_type = type(self._val)
            if ((self._val_type)(0.) <= _val <= (self._val_type)(+1.)) :
                set_constant = set()
                set_product = set()
                set_constant.add((self._val_type)(0.)) # const. for empty set
                set_product.add((self._val_type)(+1.)) # const. for product set
                set_product.add(self) # add just 1 factor to the product
                self._set_products_sum = {
                    frozenset(set_constant), # 1 & only 1 set with a constant
                    frozenset(set_product),
                }
            else :
                raise ValueError("Invalid value for fuzzy boolean")
        else : # if val is None
            self._val = None
            self._val_type = None
            self._set_products_sum = set()

    def Not(self) : # NOT(x1) = 1 - x1
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=None, dict_products_consts={}, set_products_sum=set(),
            init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=self, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __invert__(self) : # ~
        return self.Not()

    def And(self, other) : # AND(x1, x2) = x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in And method")
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self, right_operand=other,
            dict_products_consts={}, set_products_sum=set(),
            init_const_factor = (self._val_type)(+1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __and__(self, other) : # &
        return self.And(other=other)

    def Or(self, other) :  # OR(x1, x2) = x1 + x2 - x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in Or method")
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=self, dict_products_consts={},
            set_products_sum=set(), init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=other, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self, right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __or__(self, other) : # &
        return self.Or(other=other)

    def Xor(self, other) : # XOR(x1, x2) = x1 + x2 - 2 * x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in Xor method")
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=self, dict_products_consts={},
            set_products_sum=set(), init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=other, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self, right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-2.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __xor__(self, other) : # ^
        return self.Xor(other=other)

    def If(self, other) : # IF(x1, x2) = 1 - x1 + x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in If method")
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=None, dict_products_consts={}, set_products_sum=set(),
            init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=self, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self, right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __rshift__(self, other) : # >>
        return self.If(other=other)

    def Iff(self, other) : # IFF(x1, x2) = 1 - x1 - x2 + 2 * x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in Iff method")
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=None, dict_products_consts={}, set_products_sum=set(),
            init_const_factor = (self._val_type)(+1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=self, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        (dict_products_consts, set_products_sum) = Fzb._add_val(
            operand=other, dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self, right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+2.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __lshift__(self, other) : # <<
        return self.Iff(other=other)

    def evaluate(self) :
        products_sum = (self._val_type)(0.)
        for frozen_set_product in self._set_products_sum :
            product = (self._val_type)(+1.)
            for factor in frozen_set_product :
                product *= factor._val if isinstance(factor, Fzb) \
                    else factor
            products_sum += product
        self._val = max((self._val_type)(0.),
                    min((self._val_type)(+1.), products_sum))

    def value(self) :
        if self._val is None :
            self.evaluate()
        return self._val

    def __bool__(self) :
        return bool(int(round(self.value())))

    def __str__(self) :
        return str(self.value())

    def _add_prod(left_operand, right_operand, dict_products_consts,
              set_products_sum, init_const_factor) :
        for fr_set_left_product in left_operand._set_products_sum :
            for fr_set_right_product in right_operand._set_products_sum :
                set_new_product = set()
                const_factor = init_const_factor
                for factor_left in fr_set_left_product :
                    if isinstance(factor_left, Fzb) :
                        set_new_product.add(factor_left)
                    else :
                        const_factor *= factor_left
                for factor_right in fr_set_right_product :
                    if isinstance(factor_right, Fzb) :
                        set_new_product.add(factor_right)
                    else :
                        const_factor *= factor_right
                fr_set_new_product = frozenset(set_new_product)
                # check by equality, not (necessarily) by identity
                if fr_set_new_product not in set_products_sum :
                    dict_products_consts[id(fr_set_new_product)] = const_factor
                    set_products_sum.add(fr_set_new_product)
                else :
                    # simplify: combine two products with the same variables
                    for fr_set_new_product_cl in set_products_sum:
                        if fr_set_new_product == fr_set_new_product_cl :
                            dict_products_consts[id(fr_set_new_product_cl)] +=\
                                const_factor
                            break
        return (dict_products_consts, set_products_sum)

    def _add_val(operand, dict_products_consts, set_products_sum,
              init_const_factor) :
        if operand is not None :
            for fr_set_left_product in operand._set_products_sum :
                set_new_product = set()
                const_factor = init_const_factor
                for factor_left in fr_set_left_product :
                    if isinstance(factor_left, Fzb) :
                        set_new_product.add(factor_left)
                    else :
                        const_factor *= factor_left
                fr_set_new_product = frozenset(set_new_product)
                # check by equality, not (necessarily) by identity
                if fr_set_new_product not in set_products_sum :
                    dict_products_consts[id(fr_set_new_product)] = const_factor
                    set_products_sum.add(fr_set_new_product)
                else :
                    # simplify: combine two products with the same variables
                    for fr_set_new_product_cl in set_products_sum:
                        if fr_set_new_product == fr_set_new_product_cl :
                            dict_products_consts[id(fr_set_new_product_cl)] +=\
                                const_factor
                            break
        else :
            set_new_product = set()
            const_factor = init_const_factor
            fr_set_new_product = frozenset(set_new_product)
            # check by equality, not (necessarily) by identity
            if fr_set_new_product not in set_products_sum :
                dict_products_consts[id(fr_set_new_product)] = const_factor
                set_products_sum.add(fr_set_new_product)
            else :
                # simplify: combine two products with the same variables
                for fr_set_new_product_cl in set_products_sum:
                    if fr_set_new_product == fr_set_new_product_cl :
                        dict_products_consts[id(fr_set_new_product_cl)] +=\
                            const_factor
                        break
        return (dict_products_consts, set_products_sum)

    def _create(val_type, dict_products_consts, set_products_sum,
                bool_evaluate = False) :
        fzb = Fzb()
        fzb._val_type = val_type
        fzb._set_products_sum = set()
        for fr_set_new_product in set_products_sum :
            const = dict_products_consts[id(fr_set_new_product)]
            if not Fzb._is_zero(const) :
                set_new_product = set(fr_set_new_product)
                set_new_product.add(const)
                fzb._set_products_sum.add(frozenset(set_new_product))
        if bool_evaluate :
            fzb.evaluate() # may be skipped to support lazy evaluation
        return fzb

    def _is_zero(const) :
        if isinstance(const, float) :
            return -Fzb.FLT_EPS <= const <= Fzb.FLT_EPS
        elif isinstance(const, Decimal) :
            return -Fzb.DEC_EPS <= const <= Fzb.DEC_EPS
        elif isinstance(const, mpf) :
            return -Fzb.MPF_EPS <= const <= Fzb.MPF_EPS
        elif isinstance(const, Fzb) :
            return const.is_zero()
        else :
            raise TypeError("Invalid value type")

    def is_zero(self):
        if self._val_type == float :
            return -Fzb.FLT_EPS <= self.value() <= Fzb.FLT_EPS
        elif self._val_type == Decimal :
            return -Fzb.DEC_EPS <= self.value() <= Fzb.DEC_EPS
        elif self._val_type == mpf :
            return -Fzb.MPF_EPS <= self.value() <= Fzb.MPF_EPS
        else :
            raise TypeError("Invalid value type")

    def is_one(self):
        if self._val_type == float :
            return (self._val_type)(+1.) - Fzb.FLT_EPS <= self.value() <= \
                (self._val_type)(+1.) + Fzb.FLT_EPS
        elif self._val_type == Decimal :
            return (self._val_type)(+1.) - Fzb.DEC_EPS <= self.value() <= \
                (self._val_type)(+1.) + Fzb.DEC_EPS
        elif self._val_type == mpf :
            return (self._val_type)(+1.) - Fzb.MPF_EPS <= self.value() <= \
                (self._val_type)(+1.) + Fzb.MPF_EPS
        else :
            raise TypeError("Invalid value type")


def main():
    fb_flt_x = Fzb(float(   0.7566666666666666666666666666666666666))
    fb_dec_x = Fzb(Decimal("0.7566666666666666666666666666666666666"))
    fb_mpf_x = Fzb(mpf(    "0.7566666666666666666666666666666666666"))
    print(fb_flt_x.value())
    print(fb_dec_x.value())
    print(fb_mpf_x.value())

    fb_flt_y = fb_flt_x.Not()
    fb_dec_y = fb_dec_x.Not()
    fb_mpf_y = fb_mpf_x.Not()
    print(fb_flt_x.value())
    print(fb_dec_x.value())
    print(fb_mpf_x.value())
    print(fb_flt_y.value())
    print(fb_dec_y.value())
    print(fb_mpf_y.value())
    print(bool(fb_flt_x))
    print(bool(fb_dec_x))
    print(bool(fb_mpf_x))
    print(bool(fb_flt_y))
    print(bool(fb_dec_y))
    print(bool(fb_mpf_y))
    print(fb_flt_x.value())
    print(fb_dec_x.value())
    print(fb_mpf_x.value())
    print(fb_flt_y.value())
    print(fb_dec_y.value())
    print(fb_mpf_y.value())

    print()

    fb_flt_x = Fzb(float(   0.75))
    fb_flt_z = Fzb(float(   0.25))
    fb_flt_a = fb_flt_x.And(fb_flt_z)
    print("Independent inputs (x and z):")
    print(fb_flt_x.value(), "and", fb_flt_z.value(), "=", fb_flt_a.value())
    fb_flt_b = fb_flt_x.Or(fb_flt_z)
    print(fb_flt_x.value(), "or", fb_flt_z.value(), "=", fb_flt_b.value())

    fb_flt_x = Fzb(float(   0.75))
    fb_flt_z = fb_flt_x.Not()
    fb_flt_a = fb_flt_x.And(fb_flt_z)
    print("Dependent inputs (x and not(x)):")
    print(fb_flt_x.value(), "and", fb_flt_z.value(), "=", fb_flt_a.value())
    fb_flt_b = fb_flt_x.Or(fb_flt_z)
    print(fb_flt_x.value(), "or", fb_flt_z.value(), "=", fb_flt_b.value())

    fb_flt_x = Fzb(float(   0.75))
    fb_flt_z = fb_flt_x.Not()
    fb_flt_a = fb_flt_x.Not().And(fb_flt_z)
    print("Dependent inputs (not(x) and not(x)):")
    print("not(", fb_flt_x.value(), ") and", fb_flt_z.value(), "=", fb_flt_a.value())
    fb_flt_b = fb_flt_x.Not().Or(fb_flt_z)
    print("not(", fb_flt_x.value(), ") or", fb_flt_z.value(), "=", fb_flt_b.value())

    print()
    print(Fzb(0.75).Xor(Fzb(0.25)))
    print(Fzb(0.5).Xor(Fzb(0.5)))
    print(Fzb(0.75).If(Fzb(0.25)))
    print(Fzb(0.5).If(Fzb(0.5)))
    print(Fzb(0.75).Iff(Fzb(0.25)))
    print(Fzb(0.5).Iff(Fzb(0.5)))

    #from decimal import Decimal
    #from mpmath import mpf
    #from fz import Fzb

    x1 = Fzb(.75)
    x2 = Fzb(.25)
    x3 = Fzb(.75)
    print("x1 = {0:s}".format(str(x1)))
    print("x2 = {0:s}".format(str(x2)))
    print("x3 = {0:s}".format(str(x3)))
    print("not(x1) = {0:s} = {1:s}".format(str(x1.Not()), str(~x1)))
    print("and(x1, x2) = {0:s} = {1:s}".format(str(x1.And(x2)), str(x1 & x2) ))
    print("and(x1, x3) = {0:s} = {1:s}".format(str(x1.And(x3)), str(x1 & x3) ))
    print("and(x1, x1) = {0:s} = {1:s}".format(str(x1.And(x1)), str(x1 & x1) ))
    print("or(x1, x2) = {0:s} = {1:s}".format(str(x1.Or(x2)), str(x1 | x2) ))
    print("or(x1, x3) = {0:s} = {1:s}".format(str(x1.Or(x3)), str(x1 | x3) ))
    print("or(x1, x1) = {0:s} = {1:s}".format(str(x1.Or(x1)), str(x1 | x1) ))
    print("xor(x1, x2) = {0:s} = {1:s}".format(str(x1.Xor(x2)), str(x1 ^ x2) ))
    print("if(x1, x2) = {0:s} = {1:s}".format(str(x1.If(x2)), str(x1 >> x2) ))
    print("iff(x1, x2) = {0:s} = {1:s}".format(str(x1.Iff(x2)), str(x1 << x2) ))
    print("xor(and(x1, x2), and(x2, x3)) = {0:s} = {1:s}".format(
        str(x1.And(x2).Xor(x2.And(x3))), str((x1 & x2) ^ (x2 & x3))))
    print()

    x4 = Fzb(Decimal(1.) / Decimal(3.))
    x5 = Fzb(Decimal(2.) / Decimal(3.))
    print("x4 = {0:s}".format(str(x4)))
    print("x5 = {0:s}".format(str(x5)))
    print("and(x4, x5) = {0:s}".format(str(x4.And(x5))))
    print()

    x6 = Fzb(mpf(1.) / mpf(3.))
    x7 = Fzb(mpf(2.) / mpf(3.))
    print("x6 = {0:s}".format(str(x6)))
    print("x7 = {0:s}".format(str(x7)))
    print("or(x6, x7) = {0:s}".format(str(x4.Or(x5))))
    print()

    lst = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10] = \
        [Fzb(Decimal(str(round(i/10.,10)))) for i in range(11)]
    for (i,x) in enumerate(lst) :
        print("x" + str(i) + "=" + str(x), end="; ")
    # x0=0.0; x1=0.1; x2=0.2; x3=0.3; x4=0.4; x5=0.5; x6=0.6; x7=0.7; x8=0.8; x9=0.9; x10=1.0;
    print()
    print( ( (~x3 & (x5 | x1)) & (x4 | x8) | (~x9 & ~x5) ) ) # 0.385720
    print( ( (x3.Not().And(x5.Or(x1))).And(x4.Or(x8)).Or(x9.Not().And(x5.Not())) ) ) # 0.385720


if __name__ ==  '__main__':
    main()

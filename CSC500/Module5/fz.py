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
    3. 12/12/2023: Added support for conditional probabilities, and thus
                   removed assumption about independence of variables.
    4. 12/12/2023: Added support for "str" as input type for fuzzy Boolean
                   value. Added feature that allows optional conditional
                   probabilities.
    5. 12/13/2023: Refactored handling of conditional probabilities (CP). Added
                   identity support for CPs: P(A|A). Added validation of inputs
                   when CP value is set for A and B: P(A|B) = value.
    6. 01/04/2024: Added fuzzy integer multiplication feature, at least for
                   relatively small integers (bigger integers cause performance
                   issue). Added comments on how to implement fuzzy
                   factorization.

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
# When Fzb._bool_rcp is False, conditional probabilities P(A | B) are not used
# to compute P(A and B) == P(A | B) * P(B) == P(B | A) * P(A). Instead, either
# independence or identiry rules are used:
# P(A and B) = P(A) * P(B) for independent.
# P(A and B) = P(A) = P(B) for identical.
###############################################################################
# When Fzb._bool_acp is True but Fzb._bool_rcp is False, P(A | B) is used
# only when it is provided for A, or else P(A) is used.
###############################################################################
# When Fzb._bool_rcp is True, conditional probabilities P(A | B) are used.
# It means that conditional probability must be provided for each fuzzy
# operator, regardless whether it is applied to atomic or composite
# operands.
# The Bayes Theorem Formula is used to derive the symmetrical conditional
# probability, as needed:
# P(A | B) = [P(B | A) * P(A)] / P(B).
# P(B | A) = [P(A | B) * P(B)] / P(A).
# During fuzzy logic expression evaluation, the products P(A | B) * P(B) would
# be used in calculations instead of P(A) * P(B).
#
# Each instance of Fzb "A" must has a dictionary of preceeding events
# Fzb "B" mapped to conditinal probabilities values Fzb P(A | B), which are
# constructed from one of the following types: "float", Decimal", or "mpf".
# These conditinal probabilities must be provided in advance, before expression
# construction (And, Or, Xor, If, Iff). As soon as P(A | B) is provided by the
# user and is inserted into the dictionary of Fzb "A", P(B | A) is
# automatically computed and inserted to the dictionary of Fzb "B":
# P(B | A) = P(A | B) * P(B) / P(A). Evaluation of numeric values
# ("float", "Decimal", or "mpf") is done for computing new P(B | A) in this
# expression. Both P(A | B) and P(B | A) are Fzb objects, created from numbers
# ("float", "Decimal", or "mpf"). During expression building P(A | B) is used
# instead of P(A) for expanding P(A and B) = P(A | B) * P(B). Note that only
# multiplication part is affected, but not the addition part.
###############################################################################
#                   Summary of factorization algorithm:
#
# 1. The product has N bits, where N is an even positive number with
#    the minimal possible number of binary zeros on the left.
#    The product must be padded with additional N zeros at the front,
#    thus making is 2 * N bits long.
# 2. Both factors must be N bits long (not 2 * N), padded with zeros
#    (or other probabilities) at the front as needed to reach size N.
# 3. Create 2 * N vertical deques (linked lists from Collections) for
#    summations. Store these deques in a fixed-length built-in list of
#    length 2 * N. The deques are initially populated with AND(a,b),
#    where "a" and "b" are bits from different respective factors,
#    selected by cross-pairing.
# 4. The initial length of each list is from 0 to N.
# 5. The lists are processed from the right most one to the left.
# 6. Pairs of ANDs are merged with XOR (AND, AND), and the XOR results
#    are appended to the same lists.
# 7. At the same time when pairs of ANDs are merged with XORs like
#    XOR(AND, AND), the carryovers are computed as AND(AND, AND), and
#    these carryovers are appended to the next list on the left.
# 8. When only one element is left in the list, this element is moved
#    to the respective bit of the product, and the next list on the
#    left is processed, unless it is empty or non-existent.
# 9. Example with N = 6. The product is no longer than 6 bits (most
#    likely, either 5 or 6, i.e. the left-mist one is in position 5 or
#    6 from the right). Six padding zeros before the product are
#    important to keep! In the image below, each 0 means 0, while each
#    1 means a probability 0 <= p <= 1.
#            |1 1 1 1 1 1
#            |1 1 1 1 1 1
# -----------+-----------
#            |1 1 1 1 1 1
#           1|1 1 1 1 1
#         1 1|1 1 1 1
#       1 1 1|1 1 1
#     1 1 1 1|1 1
#   1 1 1 1 1|1
# -----------+-----------
# 0 0 0 0 0 0|1 1 1 1 1 1
#######################################################################


from __future__ import annotations
from decimal import Decimal, getcontext, InvalidOperation
from mpmath import mpf, mp
from sys import float_info
from collections import deque
import copy


class Fzb :

    INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT = 50 # impacts DEC_EPS and MPF_EPS
    FLT_EPS = float_info.epsilon # 2.220446049250313e-16
    # Use _generate_epsilon to update epsilons, if necessary
    DEC_EPS = Decimal("8.552847072295026067649716694884742012208630016336E-50")
    MPF_EPS = mpf("2.672764710092195646140536467151481878815196880105e-51")
    getcontext().prec = INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT
    mp.dps = INT_NUMBER_OF_DIGITS_AFTER_DECIMAL_POINT

    _bool_acp = False
    _bool_rcp = False

    # https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python
    def _generate_epsilon(self) -> float | Decimal | mpf :
        eps = (self._val_type)(+1.) # float, Decimal, or mpf
        while eps + 1 > 1:
            eps /= 2
        eps *= 2
        return eps

    def __init__(self, val = None) -> None :
        self._val = None
        self._val_type = None
        self._set_products_sum = None
        self._dict_cond = None
        if val is not None :
            if not isinstance(val, bool) and \
               not isinstance(val, float) and \
               not isinstance(val, str) and \
               not isinstance(val, Decimal) and \
               not isinstance(val, mpf) :
                raise TypeError("Invalid value type for fuzzy boolean")
            if isinstance(val, bool) :
                _val = float(val)
            elif isinstance(val, str) :
                try:
                    _val = Decimal(val)
                except InvalidOperation as e :
                    print("Unsupported string value.")
                    raise e
            else :
                _val = val
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
                self._dict_cond = {} if Fzb._bool_acp or Fzb._bool_rcp else None
            else :
                raise ValueError("Invalid value for fuzzy boolean")
        else : # if val is None
            self._val = None
            self._val_type = None
            self._set_products_sum = set()
            self._dict_cond = {} if Fzb._bool_acp or Fzb._bool_rcp else None

    def Not(self) -> "Fzb" : # NOT(x1) = 1 - x1
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

    def __invert__(self) -> "Fzb" : # ~
        return self.Not()

    def And(self, other : "Fzb") -> "Fzb" : # AND(x1, x2) = x1 * x2
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in And method")
        (dict_products_consts, set_products_sum) = Fzb._add_prod(
            left_operand=self.given(other=other), right_operand=other,
            dict_products_consts={}, set_products_sum=set(),
            init_const_factor = (self._val_type)(+1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __and__(self, other : "Fzb") -> "Fzb": # &
        return self.And(other=other)

    def Or(self, other : "Fzb") -> "Fzb" :  # OR(x1, x2) = x1 + x2 - x1 * x2
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
            left_operand=self.given(other=other), right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __or__(self, other : "Fzb") -> "Fzb" : # &
        return self.Or(other=other)

    def Xor(self, other : "Fzb") -> "Fzb" :# XOR(x1,x2) = x1 + x2 - 2 * x1 * x2
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
            left_operand=self.given(other=other), right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(-2.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __xor__(self, other : "Fzb") -> "Fzb" : # ^
        return self.Xor(other=other)

    def If(self, other : "Fzb") -> "Fzb" : # IF(x1, x2) = 1 - x1 + x1 * x2
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
            left_operand=self.given(other=other), right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+1.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __rshift__(self, other : "Fzb") -> "Fzb" : # >>
        return self.If(other=other)

    def Iff(self, other : "Fzb") -> "Fzb" : # IFF(x1,x2)=1 - x1 - x2 + 2*x1*x2
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
            left_operand=self.given(other=other), right_operand=other,
            dict_products_consts=dict_products_consts,
            set_products_sum=set_products_sum,
            init_const_factor = (self._val_type)(+2.))
        return Fzb._create(
            val_type = self._val_type,
            dict_products_consts = dict_products_consts,
            set_products_sum = set_products_sum)

    def __lshift__(self, other : "Fzb") -> "Fzb" : # <<
        return self.Iff(other=other)

    def evaluate(self) -> None :
        products_sum = (self._val_type)(0.)
        for frozen_set_product in self._set_products_sum :
            product = (self._val_type)(+1.)
            for factor in frozen_set_product :
                product *= factor._val if isinstance(factor, Fzb) \
                    else factor
            products_sum += product
        self._val = max((self._val_type)(0.),
                    min((self._val_type)(+1.), products_sum))

    def value(self) -> float | Decimal | mpf :
        if self._val is None :
            self.evaluate()
        return self._val

    def type(self) -> type :
        return self._val_type

    def __bool__(self) -> bool :
        return bool(int(round(self.value())))

    def __str__(self) -> str :
        return str(self.value())

    def _add_prod(left_operand : "Fzb",
                  right_operand : "Fzb",
                  dict_products_consts : dict,
                  set_products_sum : set,
                  init_const_factor : float | Decimal | mpf,
                  ) -> tuple :
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

    def _add_val(operand : "Fzb",
                 dict_products_consts : dict,
                 set_products_sum : set,
                 init_const_factor : float | Decimal | mpf,
                 ) -> None :
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

    def _create(val_type : float | Decimal | mpf,
                dict_products_consts : dict,
                set_products_sum : set,
                bool_evaluate : bool = False,
                ) -> "Fzb" :
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

    def _is_zero(const #: float | Decimal | mpf | "Fzb",
                 ) -> bool :
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

    def __hash__(self) -> int :
        # Without "__hash__", "__eq__" causes:
        # "TypeError: unhashable type: 'Fzb'"
        return id(self)

    def __eq__(self, other : "Fzb") -> bool :
        if self._val_type != other._val_type :
            raise TypeError("Inconsistent types in == method")
        if self is other :
            return True
        else :
            return Fzb._is_zero(
                (self.value() - other.value()) / (self._val_type)(+2.))

    def is_zero(self) -> bool :
        if self._val_type == float :
            return -Fzb.FLT_EPS <= self.value() <= Fzb.FLT_EPS
        elif self._val_type == Decimal :
            return -Fzb.DEC_EPS <= self.value() <= Fzb.DEC_EPS
        elif self._val_type == mpf :
            return -Fzb.MPF_EPS <= self.value() <= Fzb.MPF_EPS
        else :
            raise TypeError("Invalid value type")

    def is_one(self) -> bool :
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

    def conditional_on(self, other : "Fzb",
                       val # : float | Decimal | mpf | "Fzb",
                       ) -> None :
        # P(A | B) = P(self | other) = Fzb(val)
        if Fzb._bool_acp or Fzb._bool_rcp :
            if self._dict_cond is None :
                self._dict_cond = {}
            if isinstance(val, Fzb) :
                fzb_val = val
            else :
                fzb_val = Fzb(val)

            if self is other :
                if self.is_zero() :
                    if not fzb_val.is_zero() : # 0 < P(A|A) <= 1; P(A) = 0
                        raise ValueError("Invalid P(A|A)!=0 for P(A)=0")
                    else : # P(A|A) = 0; P(A) = 0
                        self._dict_cond[other] = fzb_val # overwrite if needed
                elif self.is_one() :
                    if not fzb_val.is_one() : # 0 <= P(A|A) < 1; P(A) = 1
                        raise ValueError("Invalid P(A|A)!=1 for P(A)=1")
                    else : # P(A|A) = 1; P(A) = 1
                        self._dict_cond[other] = fzb_val # overwrite if needed
                else :
                    if not fzb_val.is_one() : # 0 <= P(A|A) < 1; 0 < P(A) < 1
                        raise ValueError("Invalid P(A|A)!=1 for 0<P(A)<1")
                    else : # P(A|A) = 1; 0 < P(A) < 1
                        self._dict_cond[other] = fzb_val # overwrite if needed
            else : # self is not other
                if self.is_zero() : # A
                    if other.is_zero() : # B
                        if fzb_val.is_zero() :  # P(A)=0; P(B)=0; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                        elif fzb_val.is_one() : # P(A)=0; P(B)=0; P(A|B)=1.
                            raise ValueError("P(A|B)!=0, but P(A)=0")
                        else :                  # P(A)=0; P(B)=0; 0<P(A|B)<1.
                            raise ValueError("P(A|B)!=0, but P(A)=0")
                    elif other.is_one() : # B
                        if fzb_val.is_zero() :  # P(A)=0; P(B)=1; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # P(A)=P(A|B)=0
                        elif fzb_val.is_one() : # P(A)=0; P(B)=1; P(A|B)=1.
                            raise ValueError("P(A)=0, but P(A|B)!=0")
                        else :                  # P(A)=0; P(B)=1; 0<P(A|B)<1.
                            raise ValueError("P(A)=0, but P(A|B)!=0")
                    else : # B
                        if fzb_val.is_zero() :  # P(A)=0; 0<P(B)<1; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # P(A) ind. P(B)
                        elif fzb_val.is_one() : # P(A)=0; 0<P(B)<1; P(A|B)=1.
                            raise ValueError("P(A)=0, but P(A|B)!=0")
                        else :                  # P(A)=0; 0<P(B)<1; 0<P(A|B)<1.
                            raise ValueError("P(A)=0, but P(A|B)!=0")
                elif self.is_one() : # A
                    if other.is_zero() : # B
                        if fzb_val.is_zero() :  # P(A)=1; P(B)=0; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                        elif fzb_val.is_one() : # P(A)=1; P(B)=0; P(A|B)=1.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                        else :                  # P(A)=1; P(B)=0; 0<P(A|B)<1.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                    elif other.is_one() : # B
                        if fzb_val.is_zero() :  # P(A)=1; P(B)=1; P(A|B)=0.
                            raise ValueError("P(A)=P(B)=1, but P(A|B)!=1")
                        elif fzb_val.is_one() : # P(A)=1; P(B)=1; P(A|B)=1.
                            self._dict_cond[other] = fzb_val # fully determin.
                        else :                  # P(A)=1; P(B)=1; 0<P(A|B)<1.
                            raise ValueError("P(A)=P(B)=1, but P(A|B)!=1")
                    else : # B
                        if fzb_val.is_zero() :  # P(A)=1; 0<P(B)<1; P(A|B)=0.
                            raise ValueError("P(A)=1, but P(A|B)!=1")
                        elif fzb_val.is_one() : # P(A)=1; 0<P(B)<1; P(A|B)=1.
                            self._dict_cond[other] = fzb_val # P(A) indep P(B)
                        else :                  # P(A)=1; 0<P(B)<1; 0<P(A|B)<1.
                            raise ValueError("P(A)=1, but P(A|B)!=1")
                else : # A
                    if other.is_zero() : # B
                        if fzb_val.is_zero() :  # 0<P(A)<1; P(B)=0; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                        elif fzb_val.is_one() : # 0<P(A)<1; P(B)=0; P(A|B)=1.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                        else :                  # 0<P(A)<1; P(B)=0; 0<P(A|B)<1.
                            self._dict_cond[other] = fzb_val # P(A|B) not used
                    elif other.is_one() : # B
                        if fzb_val.is_zero() :  # 0<P(A)<1; P(B)=1; P(A|B)=0.
                            raise ValueError("P(B)=1 & P(A|B)=0, but P(A)!=0")
                        elif fzb_val.is_one() : # 0<P(A)<1; P(B)=1; P(A|B)=1.
                            raise ValueError("P(B)=1 & P(A|B)=1, but P(A)!=1")
                        else :                  # 0<P(A)<1; P(B)=1; 0<P(A|B)<1.
                            if self == fzb_val :
                                self._dict_cond[other] = fzb_val # P(A|B)=P(A)
                            else :
                                raise ValueError("P(B)=1 and P(A|B)!=P(A)")
                    else : # B
                        if fzb_val.is_zero() :  # 0<P(A)<1; 0<P(B)<1; P(A|B)=0.
                            self._dict_cond[other] = fzb_val # A, B disjoint
                        elif fzb_val.is_one() : # 0<P(A)<1; 0<P(B)<1; P(A|B)=1.
                            self._dict_cond[other] = fzb_val # A superset of B
                        else :                # 0<P(A)<1; 0<P(B)<1; 0<P(A|B)<1.
                            self._dict_cond[other] = fzb_val # general case

            # Bayes' theorem
            if self.is_zero() :
                # P(other | self) = P(other), when P(self) = 0
                other._dict_cond[self] = other
            else :
                # P(other | self) = (P(self | other) * P(other)) / P(self)
                other._dict_cond[self] = \
                    Fzb(fzb_val.value() * other.value() / self.value())

    def given(self, other : "Fzb") -> "Fzb" :
        # Return value of conditional probability
        if self is other and (
           self._dict_cond is None or self not in self._dict_cond):
            if self.is_zero() :
                return Fzb((self._val_type)(0.)) # P(A|A) = 0; P(A) = 0.
            else :
                return Fzb((self._val_type)(+1.)) # P(A|A) = 1; 0 < P(A) <= 1.
        if Fzb._bool_acp and not Fzb._bool_rcp and other in self._dict_cond :
            return self._dict_cond[other]
        elif Fzb._bool_rcp :
            return self._dict_cond[other] # Exception if not exists
        else : # other not in self._dict_cond
            return self

    def assume_independence(bool_indep : bool = True) -> None :
        Fzb.allow_conditional_probabilities(bool_acp = not bool_indep)
        Fzb.require_conditional_probabilities(bool_rcp = not bool_indep)

    def assume_independence_by_default(
            bool_indep_by_dft : bool = True) -> None :
        Fzb.allow_conditional_probabilities(bool_acp = True)
        Fzb.require_conditional_probabilities(bool_rcp = not bool_indep_by_dft)

    def allow_conditional_probabilities(bool_acp : bool = True) -> None :
        # If Fzb._bool_acp == True, value from self._dict_cond
        # (not self._val) will be used to for expression building in And, Or,
        # Xor, If, and Iff methods in each Fzb object, as long as it exist,
        # or else self._val is used.
        Fzb._bool_acp = bool_acp

    def require_conditional_probabilities(bool_rcp : bool = True) -> None :
        # If Fzb._bool_rcp == True, value from self._dict_cond
        # (not self._val) will be used to for expression building in And, Or,
        # Xor, If, and Iff methods in each Fzb object, as long as it exist,
        # or else an exception of type "InvalidOperation" is raised.
        if bool_rcp :
            Fzb._bool_acp = True
        Fzb._bool_rcp = bool_rcp

    def integer_to_fzb_list(
            int_value : int = 0, # must be non-negative
            int_min_num_bits : int = 1, # must be positive
            fzb_type : type [float, Decimal, mpf] = float,) -> list[Fzb] :
        # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        # The binary representation is padded with zeros at the front, if
        # possible or/and when necessary.
        str_bin_repr = ('{0:0'+str(int_min_num_bits)+'b}').format(int_value)
        lst_fzb_repr = [Fzb(val = fzb_type(1.) if c == '1' else fzb_type(0.)) \
                        for c in str_bin_repr]
        return lst_fzb_repr

    def fzb_list_to_floating_point(
            lst_fzb : list[Fzb]) -> float | Decimal | mpf :
        fzb_type = lst_fzb[0].type() if len(lst_fzb) > 0 else float
        result = fzb_type(0.)
        curr_factor = fzb_type(1.)
        two_factor = fzb_type(2.)
        for i in range(len(lst_fzb) - 1, -1, -1) :
            result += (lst_fzb[i].value() * curr_factor)
            curr_factor *= two_factor
        return result

    def multiply_integers(
            int_left_factor : int,
            int_right_factor : int,
            product_type : type [float, Decimal, mpf] = float,
            ) -> float | Decimal | mpf :
        (lst_fzb_left_factor, lst_fzb_right_factor, lst_fzb_product) = \
            Fzb._init_in_out_for_long_multipl_from_int(
                int_left_factor = int_left_factor,
                int_right_factor = int_right_factor,
                product_type = product_type,)
        lst_deque_columns = Fzb._init_interm_val_for_long_multipl(
            lst_fzb_left_factor = lst_fzb_left_factor,
            lst_fzb_right_factor = lst_fzb_right_factor,)
        if len(lst_fzb_product) != len(lst_deque_columns) :
            raise("Error: different lengths of product list vs. deque list.")
        lst_fzb_product = Fzb._run_long_multipl(
            lst_deque_columns = lst_deque_columns,
            lst_fzb_product = lst_fzb_product,)
        product = Fzb.fzb_list_to_floating_point(lst_fzb = lst_fzb_product)
        return product

    def multiply_fzb_lists(
            lst_fzb_left_factor : list[Fzb],
            lst_fzb_right_factor :  list[Fzb],
            ) -> list[Fzb] :
        (lst_fzb_left_factor, lst_fzb_right_factor, lst_fzb_product) = \
            Fzb._init_in_out_for_long_multipl_from_lst(
                lst_fzb_left_factor = lst_fzb_left_factor,
                lst_fzb_right_factor = lst_fzb_right_factor,)
        lst_deque_columns = Fzb._init_interm_val_for_long_multipl(
            lst_fzb_left_factor = lst_fzb_left_factor,
            lst_fzb_right_factor = lst_fzb_right_factor,)
        if len(lst_fzb_product) != len(lst_deque_columns) :
            raise("Error: different lengths of product list vs. deque list.")
        lst_fzb_product = Fzb._run_long_multipl(
            lst_deque_columns = lst_deque_columns,
            lst_fzb_product = lst_fzb_product,)
        return lst_fzb_product

    def _init_in_out_for_long_multipl_from_lst(
            lst_fzb_left_factor : list[Fzb],
            lst_fzb_right_factor :  list[Fzb],
            ) -> tuple[list[Fzb], list[Fzb], list[Fzb]]:
        product_type = lst_fzb_left_factor[0].type()
        int_factors_target_length = max(
            len(lst_fzb_left_factor), len(lst_fzb_right_factor))
        if int_factors_target_length > len(lst_fzb_left_factor) :
            lst_fzb_left_factor = [
                Fzb(val = product_type(0.)) for _ in range(
                    int_factors_target_length - len(lst_fzb_left_factor))] + \
                lst_fzb_left_factor
        if int_factors_target_length > len(lst_fzb_right_factor) :
            lst_fzb_right_factor = [
                Fzb(val = product_type(0.)) for _ in range(
                    int_factors_target_length - len(lst_fzb_right_factor))] + \
                lst_fzb_right_factor
        int_product_target_length = 2 * int_factors_target_length
        lst_fzb_product = [Fzb(val = product_type(0.)) for _ in range(
            int_product_target_length)]
        return (lst_fzb_left_factor, lst_fzb_right_factor, lst_fzb_product)

    def _init_in_out_for_long_multipl_from_int(
            int_left_factor : int,
            int_right_factor : int,
            product_type : type [float, Decimal, mpf] = float
            ) -> tuple[list[Fzb], list[Fzb], list[Fzb]]:
        lst_fzb_left_factor = Fzb.integer_to_fzb_list(
            int_value = int_left_factor, int_min_num_bits = 1,
            fzb_type = product_type)
        lst_fzb_right_factor = Fzb.integer_to_fzb_list(
            int_value = int_right_factor, int_min_num_bits = 1,
            fzb_type = product_type)
        int_factors_target_length = max(
            len(lst_fzb_left_factor), len(lst_fzb_right_factor))
        if int_factors_target_length > len(lst_fzb_left_factor) :
            lst_fzb_left_factor = [
                Fzb(val = product_type(0.)) for _ in range(
                    int_factors_target_length - len(lst_fzb_left_factor))] + \
                lst_fzb_left_factor
        if int_factors_target_length > len(lst_fzb_right_factor) :
            lst_fzb_right_factor = [
                Fzb(val = product_type(0.)) for _ in range(
                    int_factors_target_length - len(lst_fzb_right_factor))] + \
                lst_fzb_right_factor
        int_product_target_length = 2 * int_factors_target_length
        lst_fzb_product = [Fzb(val = product_type(0.)) for _ in range(
            int_product_target_length)]
        return (lst_fzb_left_factor, lst_fzb_right_factor, lst_fzb_product)

    def _init_interm_val_for_long_multipl(
            lst_fzb_left_factor : list [Fzb],
            lst_fzb_right_factor : list [Fzb],
            ) -> list[deque[Fzb]]:
        # Initialize list of deques for long (aka grade-school or standard)
        # multiplication.
        # https://en.wikipedia.org/wiki/Multiplication_algorithm
        # https://docs.python.org/3/library/collections.html#collections.deque
        if len(lst_fzb_left_factor) != len(lst_fzb_right_factor) :
            raise ValueError("Error: different lengths of factor lists.")
        lst_deque_ands = [deque() for _ in range(2 * len(lst_fzb_left_factor))]
        int_deque_index = len(lst_deque_ands) - 1
        for int_right_factor_index in range(len(
                lst_fzb_right_factor) - 1, -1, -1) :
            for int_left_factor_index in range(len(
                    lst_fzb_left_factor) - 1, -1, -1) :
                lst_deque_ands[int_deque_index].append(lst_fzb_right_factor[
                    int_right_factor_index] & lst_fzb_left_factor[
                        int_left_factor_index])
                int_deque_index -= 1
            int_deque_index += (len(lst_fzb_left_factor) - 1)
        return lst_deque_ands

    def _run_long_multipl(
            lst_deque_columns : list[deque],
            lst_fzb_product : list[Fzb],) -> list[Fzb] :
        int_deque_index = len(lst_deque_columns) - 1
        while int_deque_index >= 0 :
            deque_curr_column = lst_deque_columns[int_deque_index]
            while len(deque_curr_column) > 0 :
                if len(deque_curr_column) == 1 :
                    lst_fzb_product[int_deque_index] = \
                        deque_curr_column.popleft() # flush to product
                else : # if int_curr_deque_length >= 1
                    fzb_left = deque_curr_column.popleft()
                    fzb_right = deque_curr_column.popleft()
                    # Add with "Xor"
                    deque_curr_column.append(fzb_left ^ fzb_right)
                    if (int_deque_index - 1) >= 0 :
                        # Carryover with "And"
                        lst_deque_columns[int_deque_index - 1].append(
                            fzb_left & fzb_right)
            int_deque_index -= 1
        return lst_fzb_product

    def bitwise_probability_to_fzb_list(
            probability : float | Decimal | mpf = .5,
            int_num_bits : int = 1) -> list[Fzb] :
        lst_fzb_repr = [Fzb(val = probability) for _ in range(int_num_bits)]
        return lst_fzb_repr


class Fzi :

    def __init__(
            self,
            # iff "int", must be non-negative
            val : int | float | Decimal | mpf | list[Fzb] = int(0),
            int_min_num_bits : int = None, # 1; must be positive
            fzb_type : type [float, Decimal, mpf] = None, # float
            bool_deep_copy : bool = True,) -> None :
        self._flt_val = None
        self._lst_fzb = None
        if not isinstance(val, int) and \
           not isinstance(val, float) and \
           not isinstance(val, Decimal) and \
           not isinstance(val, mpf) and \
           not isinstance(val, list) :
            raise TypeError("Invalid value type for fuzzy integer")
        if isinstance(val, int) and val < 0 :
            raise TypeError("Invalid negative value for fuzzy integer")
        if int_min_num_bits is not None and \
           (not isinstance(int_min_num_bits, int) or \
            isinstance(int_min_num_bits, int) and int_min_num_bits < 1 ) :
            raise TypeError("Invalid non-positive minimum number of fuzzy " +
                            "bits for fuzzy integer")
        if fzb_type is not None and fzb_type is not float and \
           fzb_type is not Decimal and fzb_type is not mpf :
            raise TypeError("Invalid fuzzy bit type for fuzzy integer")
        # if isinstance(val,list), then all Fzb elems must have type fzb_type
        if isinstance(val, int) :
            self._lst_fzb = Fzb.integer_to_fzb_list(
                int_value = val,
                int_min_num_bits = 1 if int_min_num_bits is None \
                    else int_min_num_bits,
                fzb_type = float if fzb_type is None \
                    else fzb_type,)
        elif isinstance(val, float) or isinstance(val, Decimal) or \
             isinstance(val, mpf):
            self._lst_fzb = Fzb.bitwise_probability_to_fzb_list(
                    probability = val, int_num_bits = \
                        1 if int_min_num_bits is None else int_min_num_bits)
        elif isinstance(val, list) :
            if bool_deep_copy :
                # deep copy for list, but not for its Fzb elements
                self._lst_fzb = copy.deepcopy(val)
            else :
                self._lst_fzb = val
        else :
            raise TypeError("Invalid value type for fuzzy integer")

    def __mul__(self, other):
        lst_fzb = Fzb.multiply_fzb_lists(
            lst_fzb_left_factor = self._lst_fzb,
            lst_fzb_right_factor = other._lst_fzb,)
        return Fzi(
            val = lst_fzb,
            int_min_num_bits = 1,
            fzb_type = lst_fzb[0].type(),
            bool_deep_copy = True)

    def multiply_integers(
            int_left_factor : int,
            int_right_factor : int,
            product_type : type [float, Decimal, mpf] = float,
            ) -> float | Decimal | mpf :
        return Fzb.multiply_integers(
            int_left_factor = int_left_factor,
            int_right_factor = int_right_factor,
            product_type = product_type,)

    def evaluate(self) -> None :
        self._flt_val = Fzb.fzb_list_to_floating_point(lst_fzb = self._lst_fzb)

    def value(self) -> float | Decimal | mpf :
        if self._flt_val is None :
            self.evaluate()
        return self._flt_val

    def type(self) -> type :
        return self._lst_fzb[0].type()

    def __bool__(self) -> bool :
        return bool(self.value())

    def __str__(self) -> str :
        return str(self.value())

    def __repr__(self) -> str :
        return "; ".join([str(fzb.value()) for fzb in self._lst_fzb])

    def __len__(self) -> int :
        return len(self._lst_fzb)

    def __getitem__(self, key : int) -> Fzb :
        return self._lst_fzb[key]

    def __setitem__(self, key : int, val : Fzb) -> None :
        # Warning: fuzzy bits in a fuzzy integer are indexed from right to left
        # In other words, fuzzy bit with index zero is adjacent to the
        # fractional point to the right.
        if key < 0 :
            # insert at index 0
            # self._lst_fzb.insert(0, val)
            self._lst_fzb.append(val)
        elif key < len(self) :
            # set at index key
            self._lst_fzb[len(self) - key - 1] = val
        else :
            # append
            # self._lst_fzb.append(val)
            self._lst_fzb.insert(0, val)
        if self._flt_val is not None :
            self.evaluate()

def main()  -> None :

    if False :
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

    if False :
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

    if False :
        print()
        print(Fzb(0.75).Xor(Fzb(0.25)))
        print(Fzb(0.5).Xor(Fzb(0.5)))
        print(Fzb(0.75).If(Fzb(0.25)))
        print(Fzb(0.5).If(Fzb(0.5)))
        print(Fzb(0.75).Iff(Fzb(0.25)))
        print(Fzb(0.5).Iff(Fzb(0.5)))

    if False :
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

    ###########################################################################

    if False :
        # from decimal import Decimal
        # from mpmath import mpf
        # from fz import Fzb

        # Fzb.assume_independence_by_default(bool_indep_by_dft = True)
        # Fzb.allow_conditional_probabilities(bool_acp = True)
        # Fzb.assume_independence(bool_indep=False) # equivalent to the call below
        Fzb.require_conditional_probabilities(bool_rcp=True)

        print()
        print("Using conditional probabilities for independent variables:")
        P_A1 = Fzb(.75)
        P_B1 = Fzb(.25)
        P_A1_given_B1 = P_A1
        P_A1.conditional_on(other = P_B1, val = P_A1_given_B1)
        print("P(A1) = {0:s}".format(str(P_A1)))
        print("P(B1) = {0:s}".format(str(P_B1)))
        print("P(A1 | B1) = {0:s}".format(str(P_A1_given_B1)))
        print("P(B1 | A1) = {0:s}".format(str(P_B1.given(P_A1))))
        print("and(A1, B1) = {0:s} = {1:s}".format(
            str(P_A1.And(P_B1)), str(P_A1 & P_B1) ))
        print("or(A1, B1) = {0:s} = {1:s}".format(
            str(P_A1.Or(P_B1)), str(P_A1 | P_B1) ))
        print("and(B1, A1) = {0:s} = {1:s}".format(
            str(P_B1.And(P_A1)), str(P_B1 & P_A1) ))
        print("or(B1, A1) = {0:s} = {1:s}".format(
            str(P_B1.Or(P_A1)), str(P_B1 | P_A1) ))

        print()
        print("Using conditional probabilities for dependent variables:")
        P_A2 = Fzb(.75)
        P_B2 = Fzb(.25)
        P_A2_given_B2 = Fzb(.5) # not P_A2
        P_A2.conditional_on(other = P_B2, val = P_A2_given_B2)
        print("P(A2) = {0:s}".format(str(P_A2)))
        print("P(B2) = {0:s}".format(str(P_B2)))
        print("P(A2 | B2) = {0:s}".format(str(P_A2_given_B2)))
        print("P(B2 | A2) = {0:s}".format(str(P_B2.given(P_A2))))
        print("and(A2, B2) = {0:s} = {1:s}".format(
            str(P_A2.And(P_B2)), str(P_A2 & P_B2) ))
        print("or(A2, B2) = {0:s} = {1:s}".format(
            str(P_A2.Or(P_B2)), str(P_A2 | P_B2) ))
        print("and(B2, A2) = {0:s} = {1:s}".format(
            str(P_B2.And(P_A2)), str(P_B2 & P_A2) ))
        print("or(B2, A2) = {0:s} = {1:s}".format(
            str(P_B2.Or(P_A2)), str(P_B2 | P_A2) ))

    ###########################################################################

    if False :
        lst_fzb = Fzb.integer_to_fzb_list(
            int_value = 5, int_min_num_bits = 8, fzb_type = float)
        for fzb in lst_fzb :
            print(fzb.value(), end="; ")
            # print(id(fzb), end="; ")
        print()

        print()
        lst_fzb = Fzb.bitwise_probability_to_fzb_list(
            probability = float(.7777777777777777777777), int_num_bits = 8,)
        for fzb in lst_fzb :
            print(fzb.value(), end="; ")
            # print(id(fzb), end="; ")
        print()
        lst_fzb = Fzb.bitwise_probability_to_fzb_list(
            probability = Decimal(".888888888888888888888"), int_num_bits = 8,)
        for fzb in lst_fzb :
            print(fzb.value(), end="; ")
            # print(id(fzb), end="; ")
        print()
        lst_fzb = Fzb.bitwise_probability_to_fzb_list(
            probability = mpf(".6666666666666666666666666"), int_num_bits = 8,)
        for fzb in lst_fzb :
            print(fzb.value(), end="; ")
            # print(id(fzb), end="; ")
        print()
        print()

    if False :
        int_left_factor = 23
        int_right_factor = 31
        int_product = int_left_factor * int_right_factor
        print("Native (int): {0:d} * {1:d} = {2:s}".format(
            int_left_factor, int_right_factor, str(int_product)))
        flt_product = Fzb.multiply_integers(
            int_left_factor=int_left_factor, int_right_factor=int_right_factor,
            product_type = float)
        print("Fuzzy (float): {0:d} * {1:d} = {2:s}".format(
            int_left_factor, int_right_factor, str(flt_product)))
        mpf_product = Fzb.multiply_integers(
            int_left_factor=int_left_factor, int_right_factor=int_right_factor,
            product_type = mpf)
        print("Fuzzy (mpf): {0:d} * {1:d} = {2:s}".format(
            int_left_factor, int_right_factor, str(mpf_product)))
        print()

    if False :
        int_left_factor = 23
        int_right_factor = 31
        lst_fzb_left_factor = Fzb.integer_to_fzb_list(
            int_value = int_left_factor)
        lst_fzb_right_factor = Fzb.integer_to_fzb_list(
                int_value = int_right_factor)
        lst_fzb_product = Fzb.multiply_fzb_lists(
            lst_fzb_left_factor = lst_fzb_left_factor,
            lst_fzb_right_factor = lst_fzb_right_factor,)

        print()
        for fzb in lst_fzb_left_factor :
            print(fzb.value(), end="; ")
        print("* ")
        for fzb in lst_fzb_right_factor :
            print(fzb.value(), end="; ")
        print("= ")
        for fzb in lst_fzb_product :
            print(fzb.value(), end="; ")
        print("\n")
        print("{0:s} * {1:s} = {2:s}".format(
            str(Fzb.fzb_list_to_floating_point(lst_fzb_left_factor)),
            str(Fzb.fzb_list_to_floating_point(lst_fzb_right_factor)),
            str(Fzb.fzb_list_to_floating_point(lst_fzb_product)),
            ))
        print()

        lst_fzb_right_factor[2] = Fzb((lst_fzb_right_factor[2].type())(0.25))
        lst_fzb_product = Fzb.multiply_fzb_lists(
            lst_fzb_left_factor = lst_fzb_left_factor,
            lst_fzb_right_factor = lst_fzb_right_factor,)

        print()
        for fzb in lst_fzb_left_factor :
            print(fzb.value(), end="; ")
        print("* ")
        for fzb in lst_fzb_right_factor :
            print(fzb.value(), end="; ")
        print("= ")
        for fzb in lst_fzb_product :
            print(fzb.value(), end="; ")
        print("\n")
        print("{0:s} * {1:s} = {2:s}".format(
            str(Fzb.fzb_list_to_floating_point(lst_fzb_left_factor)),
            str(Fzb.fzb_list_to_floating_point(lst_fzb_right_factor)),
            str(Fzb.fzb_list_to_floating_point(lst_fzb_product)),
            ))

        lst_fzb_right_factor_alt = Fzb.integer_to_fzb_list(
                int_value = 28)
        print("\n{0:s} = ".format(str(Fzb.fzb_list_to_floating_point(
            lst_fzb_right_factor_alt))), end="")
        for fzb in lst_fzb_right_factor_alt :
            print(fzb.value(), end="; ")

    ###########################################################################
    if True :
        fzi1 = Fzi(val = 5, int_min_num_bits = 6)
        print(repr(fzi1)) # 0.0; 0.0; 0.0; 1.0; 0.0; 1.0
        print(fzi1) # 5
        print()

        fzi2 = Fzi(val = float(.777), int_min_num_bits = 6)
        print(repr(fzi2)) # 0.777; 0.777; 0.777; 0.777; 0.777; 0.777
        print(fzi2) # 48.95100000000001
        print()

        fzi3 = Fzi(val = Decimal(".888"), int_min_num_bits = 6)
        print(repr(fzi3)) # 0.888; 0.888; 0.888; 0.888; 0.888; 0.888
        print(fzi3) # 55.944
        print()

        fzi4 = Fzi(val = mpf(".999"), int_min_num_bits = 6)
        print(repr(fzi4)) # 0.999; 0.999; 0.999; 0.999; 0.999; 0.999
        print(fzi4) # 62.937
        print()

    if True :
        print()
        print()
        int_left_factor = 23
        int_right_factor = 31
        int_product = int_left_factor * int_right_factor
        print("Native (int): {0:s} * {1:s} = {2:s}".format(
            str(int_left_factor), str(int_right_factor), str(int_product)))

        fzi_flt_left_factor = Fzi(int_left_factor, fzb_type = float)
        fzi_flt_right_factor = Fzi(int_right_factor, fzb_type = float)
        fzi_flt_product = fzi_flt_left_factor * fzi_flt_right_factor
        print("Fuzzy (float): {0:s} * {1:s} = {2:s}".format(
            str(fzi_flt_left_factor), str(fzi_flt_right_factor),
            str(fzi_flt_product)))

        fzi_mpf_left_factor = Fzi(int_left_factor, fzb_type = mpf)
        fzi_mpf_right_factor = Fzi(int_right_factor, fzb_type = mpf)
        fzi_mpf_product = fzi_mpf_left_factor * fzi_mpf_right_factor
        print("Fuzzy (mpf): {0:s} * {1:s} = {2:s}".format(
            str(fzi_mpf_left_factor), str(fzi_mpf_right_factor),
            str(fzi_mpf_product)))

    if True :
        print()
        print()
        fzi_flt_left_factor = Fzi(23, fzb_type = float)
        fzi_flt_right_factor = Fzi(31, fzb_type = float)
        fzi_flt_product = fzi_flt_left_factor * fzi_flt_right_factor
        print(repr(fzi_flt_left_factor) + " *")
        print(repr(fzi_flt_right_factor) + " =")
        print(repr(fzi_flt_product))
        print()
        print("{0:s} * {1:s} = {2:s}".format(
            str(fzi_flt_left_factor),
            str(fzi_flt_right_factor),
            str(fzi_flt_product),))
        print()

        fzi_flt_right_factor[2] = Fzb(val = (fzi_flt_right_factor.type())(.25))
        fzi_flt_product = fzi_flt_left_factor * fzi_flt_right_factor
        print(repr(fzi_flt_left_factor) + " *")
        print(repr(fzi_flt_right_factor) + " =")
        print(repr(fzi_flt_product))
        print()
        print("{0:s} * {1:s} = {2:s}".format(
            str(fzi_flt_left_factor),
            str(fzi_flt_right_factor),
            str(fzi_flt_product),))
        print()
        print(str(fzi_flt_right_factor), "==", repr(fzi_flt_right_factor))

        fzi_flt_alt_right_factor = Fzi(28, fzb_type = float)
        print(str(fzi_flt_alt_right_factor), "==", repr(fzi_flt_alt_right_factor))
        print()

        fzi_flt_product = fzi_flt_left_factor * fzi_flt_alt_right_factor
        print(repr(fzi_flt_left_factor) + " *")
        print(repr(fzi_flt_alt_right_factor) + " =")
        print(repr(fzi_flt_product))
        print()
        print("{0:s} * {1:s} = {2:s}".format(
            str(fzi_flt_left_factor),
            str(fzi_flt_alt_right_factor),
            str(fzi_flt_product),))
        print()

if __name__ ==  '__main__':
    main()

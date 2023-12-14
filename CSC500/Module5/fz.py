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


from decimal import Decimal, getcontext, InvalidOperation
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


def main()  -> None :
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

###############################################################################

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


if __name__ ==  '__main__':
    main()

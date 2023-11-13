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
This module implements a class for an arbitrary precision floating point
computations. This class is a wrapper around functionality from packages
"decimal" and "mpmath". The native "float" type is supported as well.
The "main" function provides code for interactive testing of the module.

Usage: python main.py

Documentation style: https://realpython.com/documenting-python-code/

Revision History:

    1. 11/14/2023: Initial Draft.

@author: James J Johnson
@url: https://www.linkedin.com/in/james-james-johnson/
"""


from enum import Enum
from decimal import Decimal, getcontext
from mpmath import mpf, mp
# import math # can be used for math.isnan() on MyRealNumber, etc.


class MyRealNumberType(Enum) :
    """
        An Enum class representing enumarted types of floating point numbers.

        Static Attributes:
            FLOAT (int): Python native "float" type.
            DECIMAL (int): Decimal type as defined in the "decimal" package.
            MPMATH (int): "mpf" type as defined in the "mpmath" package

        Methods:
            None
    """
    FLOAT = 1
    DECIMAL = 2
    MPMATH = 3


class MyRealNumber :
    """
        A class representing a real number for an arbitrary precision floating
        point arithmetic, which is supported by two out of three predefined
        types in implementation details. The native "float" type is provided
        for completeness and compatibility. All three types may be mixed for
        binary operations between instances of MyRealNumber class.

        Attributes:
            _type (MyRealNumberType): 1 of 3 supported types MyRealNumberType.
            _value (float, Decimal, or mpf): contents (value) of the number.

        Methods:
            __init__ (MyRealNumber): Constructs a new instance of MyRealNumber.
            __str__ (str): the inverstible string representation of the value.
            __add__ (MyRealNumber): new MyRealNumber as a sum of two
                MyRealNumber instances.
            __sub__ (MyRealNumber): new MyRealNumber as a difference of two
                MyRealNumber instances.
            __mul__ (MyRealNumber): new MyRealNumber as a product of two
                MyRealNumber instances.
            __truediv__ (MyRealNumber): new MyRealNumber as a ratio of two
                MyRealNumber instances.
            __lt__ (bool): is the left MyRealNumber instance less than the
                right one.
            __eq__ (bool): are the MyRealNumber instances equal by value
                contents, but not necessarily identical
                (i.e. the same instance).
    """

    def __init__(
            self,
            str_value : str = '0.',
            enum_type : int = MyRealNumberType.FLOAT,) :
        """
                Initializes an MyRealNumber instance.

                Args:
                    str_value (str, optional): the value of the floating point
                        number as a string for parsing. Note that "__str__"
                        value can always be passed as "str_value". "NaN" is
                        supported too. The general format is like the one for
                        the native "float", but with arbitrary precision for
                        "Decimal" and "mpf" types.
                    enum_type (int): one of enumerated types from
                        MyRealNumberType.

                Returns:
                    MyRealNumber: newly created MyRealNumber.

                Raises:
                    TypeError: an unknown type of MyRealNumber is requested.
        """
        self._type = enum_type
        if enum_type == MyRealNumberType.FLOAT :
            self._value = float(str_value)
        elif enum_type == MyRealNumberType.DECIMAL :
            # https://docs.python.org/3/library/decimal.html
            self._value = Decimal(str_value)
        elif enum_type == MyRealNumberType.MPMATH :
            # https://mpmath.org/doc/current/basics.html
            self._value = mpf(str_value)
        else :
            raise TypeError("Invalid type for MyRealNumber")

    def __str__(self):
        return "{0}".format(str(self._value))

    def __add__(self, other : 'MyRealNumber') :
        if self._type == other._type :
            return self._value + other._value
        else :
            return self + MyRealNumber(
                str_value = str(other), enum_type = self._type)

    def __sub__(self, other : 'MyRealNumber') :
        if self._type == other._type :
            return self._value - other._value
        else :
            return self - MyRealNumber(
                str_value = str(other), enum_type = self._type)

    def __mul__(self, other : 'MyRealNumber') :
        if self._type == other._type :
            return self._value * other._value
        else :
            return self * MyRealNumber(
                str_value = str(other), enum_type = self._type)

    def __truediv__(self, other : 'MyRealNumber') :
        if self._type == other._type :
            try :
                return self._value / other._value
            except :
                return MyRealNumber(
                    str_value = 'NaN', enum_type = self._type)
        else :
            return self / MyRealNumber(
                str_value = str(other), enum_type = self._type)

    def __lt__(self, other : 'MyRealNumber') :
        if self._type == other._type :
            return self._value < other._value
        else :
            return self < MyRealNumber(
                str_value = str(other), enum_type = self._type)

    def __eq__(self, other : 'MyRealNumber'):
        if self._type == other._type :
            return self._value == other._value
        else :
            return self == MyRealNumber(
                str_value = str(other), enum_type = self._type)


def main():

    """
        The method is used for interactive unit testing of the class
        MyRealNumber from this module.

        Args:
            None

        Returns:
            N/A

        Raises:
            N/A
    """

    INT_NUM_DEC_DIGITS_PRECISION = 50
    getcontext().prec = INT_NUM_DEC_DIGITS_PRECISION
    mp.dps = INT_NUM_DEC_DIGITS_PRECISION
    # enum_type = MyRealNumberType.FLOAT
    # enum_type = MyRealNumberType.DECIMAL
    enum_type = MyRealNumberType.MPMATH

    ###########################################################################
    # Part 1
    ###########################################################################
    print("\nStarting Part 1: Addition and Subtraction.")
    while True :
        str_num1_value = input(
            '\nInput floating point real number for the left operand,' +
            ' or "q" to quit Part 1: ')
        if str_num1_value == 'q':
            break
        str_num2_value = input(
            'Input floating point real number for the right operand,' +
            ' or "q" to quit Part 1: ')
        if str_num2_value == 'q':
            break

        num1 = MyRealNumber(str_num1_value, enum_type)
        num2 = MyRealNumber(str_num2_value, enum_type)

        num_sum = num1 + num2
        num_difference = num1 - num2

        print()
        print(num1, "+", num2, "=", num_sum)
        print(num1, "-", num2, "=", num_difference)
    print("\nCompleted Part 1: Addition and Subtraction.")

    ###########################################################################
    # Part 2
    ###########################################################################
    print("\nStarting Part 2: Multiplication and Division.")
    while True :
        str_num1_value = input(
            '\nInput floating point real number for the left operand,' +
            ' or "q" to quit Part 2: ')
        if str_num1_value == 'q':
            break
        str_num2_value = input(
            'Input floating point real number for the right operand,' +
            ' or "q" to quit Part 2: ')
        if str_num2_value == 'q':
            break

        num1 = MyRealNumber(str_num1_value, enum_type)
        num2 = MyRealNumber(str_num2_value, enum_type)

        num_product = num1 * num2
        num_ratio = num1 / num2

        print()
        print(num1, "*", num2, "=", num_product)
        print(num1, "/", num2, "=", num_ratio)
    print("\nCompleted Part 2: Multiplication and Division.")


if __name__ ==  '__main__':
    main()

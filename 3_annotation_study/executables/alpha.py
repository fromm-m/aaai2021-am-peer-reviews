import numpy as np
import pandas as pd
from collections import defaultdict


class Unit:
    def __init__(self, rater, tag, begin, end):
        self.rater = rater
        self.tag = tag
        self.begin = begin
        self.end = end

    def __eq__(self, other):
        return (self.rater == other.rater and self.tag == other.tag
                and self.begin == other.begin and self.end == other.end)

    def __hash__(self):
        return 0

    def __str__(self):
        return ' '.join([self.rater, self.tag, str(self.begin), str(self.end)])

    def get_length(self):
        return self.end - self.begin


class Annotation:
    def __init__(self, units):
        self.raters = []
        self.tags = []
        self.units = defaultdict(list)  # rater to units
        self.begin = float('inf')
        self.length = 0
        self.overlapping = True
        self.gap = False

        for unit in units:
            if unit.rater not in self.raters:
                self.raters.append(unit.rater)
            if unit.tag not in self.tags:
                self.tags.append(unit.tag)
            self.units[unit.rater].append(unit)
            if unit.begin < self.begin:
                self.begin = unit.begin
            if unit.end > self.length:
                self.length = unit.end
        self.tags.sort()
        self.overlapping = self.check_overlapping()

    @classmethod
    def read_from_file(cls, file):

        print("Loading annotation " + file)

        units = []
        data = pd.read_csv(
            file, usecols=['rater', 'tag', 'offset_start',
                           'offset_end']).values.tolist()
        for row in data:
            units.append(Unit(row[0], row[1], row[2], row[3]))

        return cls(units)

    def check_overlapping(self):
        """
        Check whether unit overlapping exits
        :return: True, if exits; false, otherwise
        """
        for rater, units in self.units.items():
            self.units[rater] = list(set(self.units[rater]))
            self.units[rater].sort(key=lambda unit: unit.begin)
            for i in range(len(units) - 1):
                unit1 = units[i]
                unit2 = units[i + 1]
                if unit1.end > unit2.begin:
                    print("Overlapping occurs: " + str(unit1) + ' ' +
                          str(unit2))
                    return True
        return False

    def fill_gap(self, text_length, min_len=0):
        """
        Generate 'NA' units between annotated units for calculating uAlpha
        :param text_length: Total length of the text
        :param min_len: Minimal length of a non-argumentative unit
        :param tag: Tag of gap (NA or NON)
        """

        self.begin = 0
        self.length = text_length

        if not self.overlapping:
            if 'NA' not in self.tags:
                self.tags.insert(0, 'NA')
            for rater, units in self.units.items():
                for i in range(len(units) - 1):
                    unit1 = units[i]
                    unit2 = units[i + 1]
                    if unit2.begin - unit1.end > min_len:
                        self.units[rater].append(
                            Unit(rater, 'NA', unit1.end, unit2.begin))
                self.units[rater].sort(key=lambda unit: unit.begin)
                if units[0].begin > min_len:
                    self.units[rater].insert(
                        0, Unit(rater, 'NA', 0, units[0].begin))
                if self.length - units[len(units) - 1].end > min_len:
                    self.units[rater].append(
                        Unit(rater, 'NA', units[len(units) - 1].end,
                             self.length))
            self.gap = True

    def remove_gap(self, tags):
        """
        Remove gaps and their tag
        :param tags: Tag of gaps (NA or NON)
        """

        if self.gap:
            for v in self.units.values():
                for u in v:
                    if u.tag in tags:
                        v.remove(u)
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
        self.gap = False

    def merge_tag(self):
        if self.gap:
            self.remove_gap(['NA'])
        for k, v in self.units.items():
            for unit in v:
                unit.tag = 'VA'
        self.tags = ['VA']

    def get_next_unit(self, rater, unit):
        """
        Get next unit of a given rater
        :return: Found unit or None
        """
        units = self.units[rater]
        index = 0
        if unit is not None:
            index = units.index(unit) + 1
        if index < len(units):
            return units[index]
        return None

    def find_next_unit(self, rater, tag, unit):
        """
        Search for next unit with a fixed tag annotated by a given rater
        :param rater: Rater of unit
        :param tag: Tag of unit
        :param unit: Current unit
        :return: Next unit if exists; None, otherwise
        """
        units = self.units[rater]
        index = 0
        if unit is not None:
            index = units.index(unit) + 1
        for i in range(index, len(units)):
            if units[i].tag == tag:
                return units[i]
        return None

    def get_length_square(self, tag):
        """
        Compute the square of total length of all units with a fixed tag
        :param tag: Tag of unit
        :return: Square of length
        """
        length_square = 0
        for rater in self.raters:
            units = self.units[rater]
            for unit in units:
                if unit.tag == tag:
                    if tag == 'NA':
                        length_square += unit.get_length()
                    else:
                        length_square += unit.get_length()**2
        return length_square

    def get_total_length_square(self):
        """
        Compute the square of total length of all units
        :return: Square of length
        """
        total_length_square = 0
        length_square = defaultdict(float)
        for tag in self.tags:
            current = self.get_length_square(tag)
            length_square[tag] = current
            total_length_square += current
        return total_length_square, length_square

    def get_intersection(self, tag):
        """
        Computes length of intersection per tag
        :return: Intersection length
        """
        result = 0
        for i in range(len(self.raters)):
            # print("intersections for rater", i)
            r1 = self.raters[i]
            for j in range(len(self.raters)):
                if i != j:
                    r2 = self.raters[j]
                    unit1 = self.find_next_unit(r1, tag, None)
                    unit2 = self.get_next_unit(r2, None)
                    while unit1 is not None:
                        intersection = 0
                        while unit2 is not None and unit2.begin < unit1.end:
                            if unit2.end > unit1.begin:
                                intersection += intersect(
                                    unit1.begin, unit1.get_length(), tag,
                                    unit2.begin, unit2.get_length(), tag)
                            if unit2.end > unit1.end:
                                break
                            unit2 = self.get_next_unit(r2, unit2)
                        result += intersection**2
                        unit1 = self.find_next_unit(r1, tag, unit1)
        return round(result / (len(self.raters) - 1), 4)

    def get_total_intersection(self):
        """
        Compute total intersection length
        :return: Total intersection length
        """
        total = 0
        intersections = defaultdict(float)
        for tag in self.tags:
            intersection = self.get_intersection(tag)
            intersections[tag] = intersection
            total += intersection
        return total, intersections

    def slide_window(self, pos, offset, span, tag, unit, rater, category):
        """
        Slide a block till a certain position is reached
        :param pos: Current position
        :param offset: Begin of current block
        :param span: Length of current block
        :param tag: Tag of current block
        :param unit: Current unit/gap containing or after the block
        :param rater: Rater of unit
        :param category: Category of unit
        :return: Next offset, length, tag, unit
        """
        if pos == offset + span:
            if unit is not None:
                if pos == unit.begin:
                    span = unit.get_length()
                    tag = unit.tag
                    if category is None:
                        unit = self.get_next_unit(rater, unit)
                    else:
                        unit = self.find_next_unit(rater, category, unit)
                else:
                    span = unit.begin - pos
                    tag = None
            else:
                span = self.begin + self.length - pos
                tag = None
            offset = pos
        return offset, span, tag, unit

    def get_observed_matrix(self):
        """
        Compute Observed matrix
        :return: Observed matrix
        """
        observed_matrix = np.zeros((len(self.tags), len(self.tags)))

        for i in range(len(self.tags)):
            for j in range(len(self.tags)):
                result = 0
                c = self.tags[i]
                k = self.tags[j]
                for r1 in range(len(self.raters)):
                    rater1 = self.raters[r1]
                    for r2 in range(len(self.raters)):
                        if r1 != r2:
                            rater2 = self.raters[r2]
                            length1, length2 = [0] * 2
                            pos, offset1, offset2 = [self.begin] * 3
                            unit1 = self.find_next_unit(rater1, c, None)
                            unit2 = self.find_next_unit(rater2, k, None)
                            tag1 = None
                            tag2 = None
                            while pos < self.length and \
                                    (unit1 or unit2) is not None:
                                offset1, length1, tag1, unit1 = \
                                    self.slide_window(pos, offset1, length1,
                                                      tag1, unit1, rater1, c)
                                offset2, length2, tag2, unit2 = \
                                    self.slide_window(pos, offset2, length2,
                                                      tag2, unit2, rater2, k)
                                result += intersect(offset1, length1, tag1,
                                                    offset2, length2, tag2)
                                pos = min(offset1 + length1, offset2 + length2)
                observed_matrix[i][j] = result / (len(self.raters) - 1)
        return np.around(observed_matrix, decimals=4)

    def get_expected_matrix_minor(self, matrix):
        """
        Compute expected matrix without gaps
        :param matrix: Observed matrix
        :return: Expected matrix
        """
        total_units = np.sum(matrix)
        total_intersection, intersections = self.get_total_intersection()
        expected_matrix = np.zeros((len(self.tags), len(self.tags)))
        for i in range(len(self.tags)):
            for j in range(len(self.tags)):
                if i == j:
                    expected_matrix[i][j] = (np.sum(matrix, axis=1)[i]
                                             * np.sum(matrix, axis=0)[j] -
                                             intersections[self.tags[j]]) / \
                                            (total_units - total_intersection
                                             / total_units)
                else:
                    expected_matrix[i][j] = np.sum(matrix, axis=1)[i] * \
                                            np.sum(matrix, axis=0)[j] / \
                                            (total_units - total_intersection
                                             / total_units)
        return np.around(expected_matrix, decimals=4)

    def get_expected_matrix(self, matrix):
        """
        Compute expected matrix with gaps
        :param matrix: Observed matrix
        :return: Expected matrix
        """
        total_units = len(self.raters) * self.length
        total_length_square, length_square = self.get_total_length_square()
        expected_matrix = np.zeros((len(self.tags), len(self.tags)))
        for i in range(len(self.tags)):
            current = length_square[self.tags[i]]
            for j in range(len(self.tags)):
                if i == j:
                    expected_matrix[i][j] = total_units * (
                        (np.sum(matrix, axis=1)[i] * np.sum(matrix, axis=0)[j]
                         - current) / (total_units**2 - total_length_square))
                else:
                    expected_matrix[i][j] = total_units * (
                        np.sum(matrix, axis=1)[i] * np.sum(matrix, axis=0)[j] /
                        (total_units**2 - total_length_square))
        return np.around(expected_matrix, decimals=3)

    def alpha_score(self, gap=False, text_length=0, min_len=0):
        """
        Compute uAlpha or cuAplpha
        :param gap: True for uApha, false, otherwise
        :param text_length: Total text length
        :param non: If adding non tag
        :param min_len: Minimal length of non unit
        :return: Alpha score
        """

        if gap and min_len > 0:
            print("uAlpha requires min_len = 0")
            return None
        if len(self.raters) == 1:
            return 1.0
        # uAlpha
        elif gap:
            self.fill_gap(text_length, min_len)
            total_units = len(self.raters) * self.length
            # print("Calculating observed matrix for u_alpha...")
            observed = self.get_observed_matrix()
            # print("Calculating expected matrix for u_alpha...")
            expected = self.get_expected_matrix(observed)
        else:
            self.remove_gap(['NA'])
            # NON with minLength
            if min_len != 0:
                self.fill_gap(text_length, min_len)
            if len(self.tags) == 1:
                print('Annotation has only 1 tag, set cuAlpha = 1')
                return '1 tag'
            # print("Calculating observed matrix for cu_alpha...")
            observed = self.get_observed_matrix()
            total_units = np.sum(observed)
            # print("Calculating expected matrix for cu_alpha...")
            expected = self.get_expected_matrix_minor(observed)

        d_o = total_units - np.sum(observed.diagonal())
        d_e = total_units - np.sum(expected.diagonal())
        if d_e == 0:
            print("Intersection only in one label, set cuAlpha = 1")
            return 'de=0'
        else:
            return np.around(1 - d_o / d_e, decimals=3)


def intersect(s1, l1, t1, s2, l2, t2):
    """
    Compute length of intersection of two units
    :param s1: Start position of unit1
    :param l1: Length of unit1
    :param t1: Tag of unit1
    :param s2: Start position of unit2
    :param l2: Length of position2
    :param t2: Tag of position2
    :return: Length of intersection
    """
    if t1 is not None and t2 is not None:
        if s1 > s2 and s1 + l1 > s2 + l2:
            return s2 + l2 - s1
        elif s1 < s2 and s1 + l1 < s2 + l2:
            return s1 + l1 - s2
        elif s1 <= s2 and l1 > l2:
            return l2
        elif s1 >= s2 and l1 <= l2:
            return l1
    else:
        return 0

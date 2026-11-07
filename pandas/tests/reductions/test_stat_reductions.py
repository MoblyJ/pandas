"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""

from decimal import Decimal
from fractions import Fraction
import inspect
import re

import numpy as np
import pytest

from pandas.errors import Pandas4Warning

import pandas as pd
import pandas._testing as tm


class TestDatetimeLikeStatReductions:
    def test_dt64_mean(self, tz_naive_fixture, index_or_series_or_array):
        tz = tz_naive_fixture

        dti = pd.date_range("2001-01-01", periods=11, tz=tz)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data

        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06", tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp("2001-01-06", tz=tz)

        # dtarr[-2] will be the first date 2001-01-1
        dtarr[-2] = pd.NaT

        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp("2001-01-06 07:12:00", tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize("freq", ["s", "h", "D", "W", "B"])
    def test_period_mean(self, index_or_series_or_array, freq):
        # GH#24757
        dti = pd.date_range("2001-01-01", periods=11)
        # shuffle so that we are not just working with monotone-increasing
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])

        warn = FutureWarning if freq == "B" else None
        msg = r"PeriodDtype\[B\] is deprecated"
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = index_or_series_or_array(parr)
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

        # parr[-2] will be the first date 2001-01-1
        parr[-2] = pd.NaT

        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean()
        with pytest.raises(TypeError, match="ambiguous"):
            obj.mean(skipna=True)

    def test_td64_mean(self, index_or_series_or_array):
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], "m8[D]")
        tdi = pd.TimedeltaIndex(m8values).as_unit("ns")

        tdarr = tdi._data
        obj = index_or_series_or_array(tdarr, copy=False)

        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected

        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT

        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()

        # exact equality fails by 1 nanosecond
        assert result2.round("us") == (result * 11.0 / 10).round("us")


class TestSeriesStatReductions:
    # Note: the name TestSeriesStatReductions indicates these tests
    #  were moved from a series-specific test file, _not_ that these tests are
    #  intended long-term to be series-specific

    def _check_stat_op(
        self, name, alternate, string_series_, check_objects=False, check_allna=False
    ):
        with pd.option_context("use_bottleneck", False):
            f = getattr(pd.Series, name)

            # add some NaNs
            string_series_[5:15] = np.nan

            # mean, idxmax, idxmin, min, and max are valid for dates
            if name not in ["max", "min", "mean", "median", "std"]:
                ds = pd.Series(pd.date_range("1/1/2001", periods=10))
                msg = f"does not support operation '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)

            # skipna or no
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))

            # check the result is correct
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))

            allna = string_series_ * np.nan

            if check_allna:
                assert np.isnan(f(allna))

            # dtype=object with None, it works!
            s = pd.Series([1, 2, 3, None, 5])
            f(s)

            # GH#2888
            items = [0]
            items.extend(range(2**40, 2**40 + 1000))
            s = pd.Series(items, dtype="int64")
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))

            # check date range
            if check_objects:
                s = pd.Series(pd.bdate_range("1/1/2000", periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp

            # check on string data
            if name not in ["sum", "min", "max"]:
                with pytest.raises(TypeError, match=None):
                    f(pd.Series(list("abc")))

            # Invalid axis.
            msg = "No axis named 1 for object type Series"
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)

            if "numeric_only" in inspect.getfullargspec(f).args:
                # only the index is string; dtype is float
                f(string_series_, numeric_only=True)

    def test_sum(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("sum", np.sum, string_series, check_allna=False)

    def test_mean(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("mean", np.mean, string_series)

    def test_median(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("median", np.median, string_series)

        # test with integers, test failure
        int_ts = pd.Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("prod", np.prod, string_series)

    def test_min(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("min", np.min, string_series, check_objects=True)

    def test_max(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        self._check_stat_op("max", np.max, string_series, check_objects=True)

    def test_var_std(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        datetime_series = pd.Series(
            np.arange(10, dtype=np.float64),
            index=pd.date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1)
        self._check_stat_op("std", alt, string_series)

        alt = lambda x: np.var(x, ddof=1)
        self._check_stat_op("var", alt, string_series)

        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)

        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self):
        string_series = pd.Series(range(20), dtype=np.float64, name="series")
        datetime_series = pd.Series(
            np.arange(10, dtype=np.float64),
            index=pd.date_range("2020-01-01", periods=10),
            name="ts",
        )

        alt = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op("sem", alt, string_series)

        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(
            len(datetime_series.values)
        )
        tm.assert_almost_equal(result, expected)

        # 1 - element series with ddof=1
        s = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self, using_python_scalars):
        # GH 62864 - returns NaN on degenerate distribution
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = pd.Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op("skew", alt, string_series)

        # test corner cases, skew() returns NaN unless there's at least 3
        # values
        min_N = 3
        for i in range(1, min_N + 1):
            s = pd.Series(np.ones(i))
            df = pd.DataFrame(np.ones((i, i)))
            assert np.isnan(s.skew())
            assert np.isnan(df.skew()).all()
            if using_python_scalars:
                assert type(s.skew()) == float
            else:
                assert isinstance(s.skew(), np.float64)  # GH53482

    def test_kurt(self):
        sp_stats = pytest.importorskip("scipy.stats")

        string_series = pd.Series(range(20), dtype=np.float64, name="series")

        alt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op("kurt", alt, string_series)

    def test_kurt_corner(self, using_python_scalars):
        # GH 62864 - returns NaN on degenerate distribution
        # test corner cases, kurt() returns NaN unless there's at least 4
        # values
        min_N = 4
        for i in range(1, min_N + 1):
            s = pd.Series(np.ones(i))
            df = pd.DataFrame(np.ones((i, i)))
            assert np.isnan(s.kurt())
            assert np.isnan(df.kurt()).all()
            if using_python_scalars:
                assert type(s.kurt()) == float
            else:
                assert isinstance(s.kurt(), np.float64)  # GH53482


@pytest.mark.parametrize(
    "opname",
    [
        "std",
        "sem",
        "var",
        "skew",
        "kurt",
    ],
)
@pytest.mark.parametrize(
    "loc, scale",
    [
        (42.0, 0.0),  # GH 62864
        (0.0, 1e-18),  # GH 62946
        (0.0, 1.0),
        (0.0, 1e18),
    ],
)
def test_reduction_consistency(opname, loc, scale):
    rng = np.random.default_rng(2)
    values = rng.normal(loc=loc, scale=scale, size=(20,))
    grps = [0] * 20
    s = pd.Series(values)
    result_series = getattr(s, opname)()

    result_gb = getattr(s.groupby(grps), opname)().iloc[0]
    tm.assert_almost_equal(result_series, result_gb)

    result_window = getattr(s.rolling(20), opname)().iloc[-1]
    tm.assert_almost_equal(result_series, result_window)

    result_frame = getattr(pd.DataFrame(s), opname)().iloc[0]
    tm.assert_almost_equal(result_series, result_frame)


@pytest.mark.parametrize("opname", ["var", "std", "sem"])
def test_stat_nobs_equal_ddof(opname):
    s = pd.Series([1.0, 2.0, 3.0])
    ddof = 3

    result_series = getattr(s, opname)(ddof=ddof)
    assert np.isnan(result_series)

    result_gb = getattr(s.groupby([0, 0, 0]), opname)(ddof=ddof).iloc[0]
    assert np.isnan(result_gb)

    result_window = getattr(s.rolling(3, min_periods=3), opname)(ddof=ddof).iloc[2]
    assert np.isnan(result_window)


# ---------------------------------------------------------------------------
# object dtype
#
# GH#34671, GH#36703, GH#44008: a statistic over object-dtype data behaves
# exactly as it does over the numeric dtype the values represent, and rejects
# anything that is not a number, the same way for every statistic.

OBJECT_STATS = ["mean", "median", "var", "std", "sem", "skew", "kurt"]


@pytest.mark.parametrize("opname", OBJECT_STATS)
@pytest.mark.parametrize(
    "data, numeric_dtype",
    [
        ([1, 2, 3, 5], "int64"),
        ([1.0, 2.0, 3.0, 5.0], "float64"),
        ([1.0, None, 3.0, 5.0], "float64"),
        ([True, False, True, True], "float64"),
        ([1.0, True, 3.0, 5.0], "float64"),
        ([Decimal(1), Decimal(2), Decimal(3), Decimal(5)], "float64"),
        ([1 + 2j, 3 + 4j, 5j, 1j], "complex128"),
    ],
)
def test_object_stat_matches_numeric_dtype(opname, data, numeric_dtype):
    obj = pd.Series(data, dtype=object)
    expected_obj = pd.Series(np.array(data, dtype=numeric_dtype))

    if numeric_dtype == "complex128" and opname in ["skew", "kurt"]:
        # GH#43770 the imaginary part would be discarded
        msg = f"reduction operation '{opname}' not allowed for this dtype"
        for ser in [obj, expected_obj]:
            with pytest.raises(TypeError, match=msg):
                getattr(ser, opname)()
        return

    warn = (
        Pandas4Warning if numeric_dtype == "complex128" and opname == "median" else None
    )
    with tm.assert_produces_warning(warn, match="deprecated on complex data"):
        expected = getattr(expected_obj, opname)()
        result = getattr(obj, opname)()

    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize("opname", OBJECT_STATS)
@pytest.mark.parametrize(
    "data, match",
    [
        (["1", "2", "3"], "'1'"),
        ([1.0, "2", 3.0], "'2'"),
        (["a", "b", "c"], "'a'"),
        ([b"1", b"2", b"3"], "b'1'"),
        ([np.datetime64("2020-01-01"), np.datetime64("2020-01-03")], "datetime64"),
        ([np.timedelta64(1, "D"), np.timedelta64(3, "D")], "timedelta64"),
        ([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")], "Timestamp"),
        ([pd.Timedelta(days=1), pd.Timedelta(days=3)], "Timedelta"),
    ],
)
def test_object_stat_raises_on_non_numbers(opname, data, match):
    obj = pd.Series(data, dtype=object)
    with pytest.raises(TypeError, match=f"Could not convert .*{match}"):
        getattr(obj, opname)()


@pytest.mark.parametrize("opname", OBJECT_STATS)
def test_object_stat_matches_frame_axis(opname):
    # the same values reduced column-wise and row-wise agree
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 5.0], "b": [4.0, 8.0, 1.0, 2.0]}).astype(
        object
    )
    expected = getattr(df.astype("float64"), opname)(axis=1)
    result = getattr(df, opname)(axis=1)
    # the reduction of an object block is still wrapped as object dtype
    tm.assert_series_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("opname", OBJECT_STATS)
def test_object_none_does_not_warn_complex(opname):
    # missing values are filled with NaN rather than routed through complex128
    obj = pd.Series([1.0, None, 2.0, 4.0], dtype=object)
    with tm.assert_produces_warning(None):
        getattr(obj, opname)()


@pytest.mark.parametrize("skipna", [True, False])
def test_object_mean_of_ndarrays(skipna):
    # regression in 3.1.0: elements that are equal-length numeric ndarrays are
    #  averaged elementwise, as sum() still does
    ser = pd.Series([np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5, 9])])
    expected = np.array([3.0, 5.0])
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)

    result = ser.to_frame().mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.iloc[0], expected)

    result = ser.groupby([1, 1, 2]).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], np.array([2.0, 3.0]))
    tm.assert_numpy_array_equal(result.loc[2], np.array([5.0, 9.0]))


def _object_series(values):
    # build element by element so numpy does not stack equal-length ndarrays
    arr = np.empty(len(values), dtype=object)
    for i, val in enumerate(values):
        arr[i] = val
    return pd.Series(arr)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data, expected",
    [
        ([np.array([1.0, 2.0]), 3.0], np.array([2.0, 2.5])),
        ([np.array([1.0, 2.0]), np.array([3.0, 4.0]), 5], np.array([3.0, 11 / 3])),
        ([np.array([1.0, 2.0]), np.float64(3.0)], np.array([2.0, 2.5])),
        ([np.array([1.0, 2.0]), np.int64(3)], np.array([2.0, 2.5])),
        ([np.array([1.0, 2.0]), True], np.array([1.0, 1.5])),
        ([np.array([1.0, 2.0]), np.bool_(True)], np.array([1.0, 1.5])),
        ([np.array([1.0, 2.0]), 1 + 1j], np.array([1 + 0.5j, 1.5 + 0.5j])),
        # broadcast against each other as numpy does
        ([np.array([1.0, 2.0]), np.array([3.0])], np.array([2.0, 2.5])),
        ([np.eye(2), np.ones((2, 2))], np.array([[1.0, 0.5], [0.5, 1.0]])),
    ],
)
def test_object_mean_of_ndarrays_and_scalars(data, expected, skipna):
    # numeric scalars mixed in with ndarrays broadcast against them, as in 2.x
    ser = _object_series(data)
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)

    result = ser.groupby([1] * len(ser)).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)


@pytest.mark.parametrize("na", [None, np.nan, pd.NA, pd.NaT])
def test_object_mean_of_ndarrays_with_na(na):
    ser = _object_series([np.array([1.0, 2.0]), na, np.array([3.0, 6.0]), 4.0])
    tm.assert_numpy_array_equal(ser.mean(), np.array([8 / 3, 4.0]))
    # every NA flavour propagates as NaN rather than raising on the arithmetic
    tm.assert_numpy_array_equal(ser.mean(skipna=False), np.array([np.nan, np.nan]))


@pytest.mark.parametrize(
    "data, match",
    [
        ([[1, 2], [3, 4]], r"\[1, 2\]"),
        ([np.array(["1", "2"]), np.array(["3", "4"])], r"array\(\['1', '2'\]"),
        ([np.array([1.0, 2.0]), "x"], "'x'"),
        ([np.array([1.0, 2.0]), [3.0, 4.0]], r"\[3\.0, 4\.0\]"),
        ([np.array([1.0, 2.0]), Decimal(3)], r"Decimal\('3'\)"),
        ([np.array([1.0, 2.0]), pd.Timestamp("2020-01-01")], "Timestamp"),
    ],
)
def test_object_mean_of_ndarrays_raises_on_non_numbers(data, match):
    # the message names the offending element, not the first ndarray
    ser = _object_series(data)
    with pytest.raises(TypeError, match=f"Could not convert {match}.* to numeric"):
        ser.mean()


def test_object_mean_of_ndarrays_shape_mismatch():
    # numpy's own broadcasting error, as in 2.x
    ser = _object_series([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="could not be broadcast"):
        ser.mean()


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("dtype", ["int64", "int32", "uint8"])
def test_object_mean_of_integer_ndarrays(dtype, skipna):
    # integer elements are averaged in float64 rather than overflowing
    ser = _object_series(
        [np.array([200, 100], dtype=dtype), np.array([100, 250], dtype=dtype)]
    )
    expected = np.array([150.0, 175.0])
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)
    result = ser.groupby([1, 1]).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_object_mean_of_float_ndarrays(dtype, skipna):
    # a NaN inside an element is part of its value; skipna only skips elements
    ser = _object_series(
        [np.array([1.5, np.nan], dtype=dtype), np.array([3.5, 4.0], dtype=dtype)]
    )
    expected = np.array([2.5, np.nan])
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)
    result = ser.groupby([1, 1]).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data, expected",
    [
        ([[1, 2], [3, 4]], np.array([2.0, 3.0])),
        ([[1.5, 2.0], [3.5, 4.0]], np.array([2.5, 3.0])),
        ([[1.5, 2.0], [Decimal(3), Decimal(4)]], np.array([2.25, 3.0])),
        # an NA inside an element is NaN there, like a NaN in a float element
        ([[1.0, None], [3.0, 4.0]], np.array([2.0, np.nan])),
        ([[1.0, pd.NA], [3.0, 4.0]], np.array([2.0, np.nan])),
    ],
)
def test_object_mean_of_object_dtype_ndarrays(data, expected, skipna):
    # regression in 3.1.0: object-dtype elements holding numbers are averaged
    #  like their numeric equivalents
    ser = _object_series([np.array(val, dtype=object) for val in data])
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)
    result = ser.groupby([1, 1]).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data, expected",
    [
        ([np.array([1, 2]), np.array([3.5, 4.5])], np.array([2.25, 3.25])),
        ([np.array([1, 2]), np.array([3, 4], dtype=object)], np.array([2.0, 3.0])),
        (
            [np.array([1.0, 2.0]), np.array([3, 4], dtype=object), 5],
            np.array([3.0, 11 / 3]),
        ),
    ],
)
def test_object_mean_of_mixed_dtype_ndarrays(data, expected, skipna):
    # elements of different dtypes, and scalars, combine as numpy would
    ser = _object_series(data)
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)
    result = ser.groupby([1] * len(ser)).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)


@pytest.mark.parametrize("na", [None, np.nan, pd.NA, pd.NaT])
@pytest.mark.parametrize("dtype", ["int64", "float64", object])
def test_object_mean_of_ndarrays_by_dtype_with_na(dtype, na):
    ser = _object_series(
        [np.array([1, 2], dtype=dtype), na, np.array([3, 6], dtype=dtype)]
    )
    tm.assert_numpy_array_equal(ser.mean(), np.array([2.0, 4.0]))
    tm.assert_numpy_array_equal(ser.mean(skipna=False), np.array([np.nan, np.nan]))
    result = ser.groupby([1, 1, 1]).mean()
    tm.assert_numpy_array_equal(result.loc[1], np.array([2.0, 4.0]))


@pytest.mark.parametrize(
    "data, match",
    [
        ([np.array(["1", "2"], dtype=object)] * 2, "'1'"),
        ([np.array([1, 2]), np.array([3, "x"], dtype=object)], "'x'"),
        ([np.array([1.0, pd.Timestamp("2020-01-01")], dtype=object)] * 2, "Timestamp"),
    ],
)
def test_object_mean_of_object_dtype_ndarrays_raises(data, match):
    # strings inside an element are no more convertible than in a column
    ser = _object_series(data)
    with pytest.raises(TypeError, match=f"Could not convert {match}.* to numeric"):
        ser.mean()


def test_object_mean_of_bool_ndarrays():
    # elementwise bool addition is logical or, so sum in float64 instead
    ser = _object_series([np.array([True, False]), np.array([True, True])])
    tm.assert_numpy_array_equal(ser.mean(), np.array([1.0, 0.5]))


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int64"),
        ([200, 100, 250], "uint8"),
        ([1.5, np.nan, 4.0], "float64"),
        ([1.5, 2.5, 4.0], "float32"),
        ([True, False], "bool"),
        ([1, None, 4], "Int64"),
        ([1, 2.5, None], object),
        ([Decimal("1.5"), Decimal("2.5")], object),
        ([1 + 2j, 3.0], object),
    ],
)
def test_mean_numeric_input_skips_ndarray_fallback(monkeypatch, data, dtype):
    # the ndarray-element fallback only runs once the numeric cast has failed
    from pandas.core import nanops

    def fail(values):
        raise AssertionError("fallback reached")

    monkeypatch.setattr(nanops, "_numeric_ndarray_elements", fail)
    ser = pd.Series(data, dtype=dtype)
    ser.mean()
    ser.mean(skipna=False)
    ser.groupby([1] * len(ser)).mean()


def test_mean_ndarray_elements_use_fallback(monkeypatch):
    from pandas.core import nanops

    orig = nanops._numeric_ndarray_elements
    calls = []

    def spy(values):
        calls.append(values)
        return orig(values)

    monkeypatch.setattr(nanops, "_numeric_ndarray_elements", spy)
    ser = _object_series([np.array([1, 2]), np.array([3, 4], dtype=object), None])
    tm.assert_numpy_array_equal(ser.mean(), np.array([2.0, 3.0]))
    assert len(calls) == 1


class _Unsupported:
    def __repr__(self) -> str:
        return "_Unsupported()"


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "invalid, match",
    [
        # strings, including numeric-looking ones, are never parsed implicitly
        ("x", "'x'"),
        ("", "''"),
        (np.str_("x"), r"np\.str_\('x'\)"),
        (b"x", "b'x'"),
        ("3", "'3'"),
        ("3.5", r"'3\.5'"),
        ("nan", "'nan'"),
        ("1e3", "'1e3'"),
        (b"3", "b'3'"),
        (np.array(["1", "2"]), r"array\(\['1', '2'\]"),
        (np.array([b"1", b"2"]), r"array\(\[b'1', b'2'\]"),
        (np.array("3"), r"array\('3'"),
        (np.array([1, "2"], dtype=object), "'2'"),
        # dictionaries and other containers
        ({}, r"\{\}"),
        ({"a": 1}, r"\{'a': 1\}"),
        (np.array([{}, {}], dtype=object), r"\{\}"),
        ([3.0, 4.0], r"\[3\.0, 4\.0\]"),
        ((3.0, 4.0), r"\(3\.0, 4\.0\)"),
        ({3}, r"\{3\}"),
        (pd.Series([3.0, 4.0]), r"Series 0 3\.0 1 4\.0 dtype: float64"),
        # unsupported objects
        (_Unsupported(), r"_Unsupported\(\)"),
        (len, "<built-in function len>"),
        (pd.Timestamp("2020-01-01"), "Timestamp"),
        (pd.Timedelta("1D"), "Timedelta"),
        (pd.Period("2020"), "Period"),
        (pd.Interval(0, 1), "Interval"),
        (np.datetime64("2020-01-01"), r"np\.datetime64\('2020-01-01'\)"),
        (np.array(["2020-01-01"], dtype="M8[D]"), r"array\(\['2020-01-01'\]"),
        (np.array([1, 2], dtype="m8[D]"), r"array\(\[1, 2\], dtype='timedelta64"),
        (
            np.array([(1, 2.0)], dtype=[("x", "i4"), ("y", "f8")]),
            r"array\(\[\(1, 2\.\)\]",
        ),
        (Decimal(3), r"Decimal\('3'\)"),
    ],
)
@pytest.mark.parametrize("position", [0, 1, -1])
def test_object_mean_of_ndarrays_raises_on_invalid(invalid, match, position, skipna):
    # invalid data mixed in with numeric ndarrays raises TypeError naming the
    #  offending element, wherever it sits and whatever NA precedes it
    data = [np.array([1.0, 2.0]), None, np.array([3.0, 4.0])]
    data.insert(position if position >= 0 else len(data), invalid)
    ser = _object_series(data)
    msg = f"Could not convert {match}.* to numeric"
    with pytest.raises(TypeError, match=msg):
        ser.mean(skipna=skipna)
    with pytest.raises(TypeError, match=msg):
        ser.to_frame().mean(skipna=skipna)
    with pytest.raises(TypeError, match="agg function failed") as err:
        ser.groupby([1] * len(ser)).mean(skipna=skipna)
    assert re.search(msg, str(err.value.__cause__))


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data, expected",
    [
        ([np.array([1.0, 2.0]), Fraction(1, 2)], np.array([0.75, 1.25])),
        ([np.array([1, 2]), Fraction(1, 2)], np.array([0.75, 1.25])),
        ([np.array([1.0, 2.0]), Fraction(1, 3)], np.array([2 / 3, 7 / 6])),
        (
            [np.array([1.0, 2.0]), Fraction(1, 2), np.array([3.0, 4.0])],
            np.array([1.5, 13 / 6]),
        ),
        ([np.array([1, 2], dtype=object), Fraction(1, 2)], np.array([0.75, 1.25])),
        (
            [np.array([Fraction(1, 2), 1.5], dtype=object), np.array([3.0, 4.0])],
            np.array([1.75, 2.75]),
        ),
    ],
)
def test_object_mean_of_ndarrays_and_fractions(data, expected, skipna):
    # Fraction is a real number like any other; 1.5.3 and 2.2.3 averaged it
    ser = _object_series(data)
    tm.assert_numpy_array_equal(ser.mean(skipna=skipna), expected)
    result = ser.groupby([1] * len(ser)).mean(skipna=skipna)
    tm.assert_numpy_array_equal(result.loc[1], expected)
    tm.assert_numpy_array_equal(ser.to_frame().mean(skipna=skipna).iloc[0], expected)


def test_object_mean_of_ndarrays_and_fraction_with_na():
    ser = _object_series([np.array([1.0, 2.0]), Fraction(1, 2), None])
    tm.assert_numpy_array_equal(ser.mean(), np.array([0.75, 1.25]))
    tm.assert_numpy_array_equal(ser.mean(skipna=False), np.array([np.nan, np.nan]))


@pytest.mark.parametrize("decimal", [Decimal("0.5"), Decimal(3)])
def test_object_mean_of_ndarrays_and_fraction_still_rejects_decimal(decimal):
    # Decimal refuses to mix with float, as it did in 1.5.3 and 2.2.3
    ser = _object_series([np.array([1.0, 2.0]), Fraction(1, 2), decimal])
    with pytest.raises(TypeError, match=r"Could not convert Decimal\("):
        ser.mean()


@pytest.mark.parametrize(
    "data",
    [
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        [np.array([1.5, 2.0], dtype="float32"), np.array([3.5, 4.0], dtype="float32")],
        [np.array([1 + 1j, 2j]), 3.0],
        [np.array([1, 2]), np.array([3, 4])],
        [np.array([1.0, 2.0]), 3, 2.5, np.float32(1.5)],
    ],
)
def test_ndarray_elements_passed_through(data):
    # elements elementwise addition already handles are not converted
    from pandas.core import nanops

    values = _object_series(data).to_numpy()
    result = nanops._numeric_ndarray_elements(values)
    for res, val in zip(result, values, strict=True):
        assert res is val


@pytest.mark.parametrize(
    "data, expected",
    [
        # bool/integer elements are summed exactly in int64, not float64
        ([np.array([200, 100], dtype="uint8"), 7], ["int64", int]),
        ([np.array([True, False]), np.bool_(True)], ["int64", int]),
        ([np.array([1, 2], dtype="int32"), np.int8(3)], ["int64", int]),
        # unless they could overflow it
        ([np.array([2**62]), np.array([2**62])], ["float64", "float64"]),
        ([np.array([1, 2]), 2**70], ["float64", float]),
        # Fraction has no ndarray arithmetic and float16 overflows early
        ([np.array([1.0]), Fraction(1, 2)], ["float64", float]),
        ([np.array([1.0], dtype="float16"), np.float16(2)], ["float64", np.float64]),
    ],
)
def test_ndarray_elements_converted(data, expected):
    from pandas.core import nanops

    result = nanops._numeric_ndarray_elements(_object_series(data).to_numpy())
    for res, exp in zip(result, expected, strict=True):
        if isinstance(exp, str):
            assert res.dtype == exp
        else:
            assert type(res) is exp


@pytest.mark.parametrize(
    "data, expected",
    [
        # summed exactly; float64 would drop each +1 against 2**53
        (
            [np.array([2**53]), np.array([1]), np.array([1])],
            np.array([(2**53 + 2) / 3]),
        ),
        ([np.array([2**62, 1]), np.array([2**62, 1])], np.array([2.0**62, 1.0])),
        ([np.array([2**64 - 1], dtype="uint64")] * 2, np.array([2.0**64 - 1])),
        ([np.array([1, 2], dtype="uint8"), -3], np.array([-1.0, -0.5])),
        (
            [np.uint8(200), np.uint8(100), np.array([0.0, 0.0])],
            np.array([100.0, 100.0]),
        ),
        ([np.bool_(True), np.bool_(True), np.array([0, 0])], np.array([2 / 3, 2 / 3])),
        ([np.float16(60000), np.float16(60000), np.array([0.0])], np.array([40000.0])),
    ],
)
def test_object_mean_of_ndarrays_integer_overflow(data, expected):
    # 2.2.3 wrapped around, warned or raised OverflowError on each of these
    ser = _object_series(data)
    with tm.assert_produces_warning(None):
        result = ser.mean()
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "na",
    [
        None,
        np.nan,
        float("nan"),
        np.float32("nan"),
        pd.NA,
        pd.NaT,
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
        Decimal("NaN"),
    ],
)
@pytest.mark.parametrize("position", [0, 1, -1])
def test_object_mean_of_ndarrays_skipna(na, position):
    # skipna=False does not raise on an NA: like a float column, the result is
    #  NaN.  1.5.3 and 2.2.3 raised for every NA but np.nan
    data = [np.array([1.0, 2.0]), np.array([3.0, 6.0])]
    data.insert(position if position >= 0 else len(data), na)
    ser = _object_series(data)
    tm.assert_numpy_array_equal(ser.mean(skipna=True), np.array([2.0, 4.0]))
    tm.assert_numpy_array_equal(ser.mean(skipna=False), np.array([np.nan, np.nan]))
    result = ser.groupby([1] * len(ser)).mean(skipna=False)
    tm.assert_numpy_array_equal(result.loc[1], np.array([np.nan, np.nan]))


@pytest.mark.parametrize(
    "data, expected_skipna, expected_no_skipna",
    [
        # a NaN *inside* an element is a value: it only affects its position
        (
            [np.array([1.0, np.nan]), np.array([3.0, 6.0])],
            np.array([2.0, np.nan]),
            np.array([2.0, np.nan]),
        ),
        (
            [np.array([1.0, None], dtype=object), np.array([3.0, 6.0])],
            np.array([2.0, np.nan]),
            np.array([2.0, np.nan]),
        ),
        # several NAs are each skipped
        (
            [np.array([1.0, 2.0]), None, np.nan, pd.NA, np.array([3.0, 6.0])],
            np.array([2.0, 4.0]),
            np.array([np.nan, np.nan]),
        ),
        # an element on its own with an NA
        ([np.array([1.0, 2.0]), None], np.array([1.0, 2.0]), np.array([np.nan] * 2)),
        # numeric scalars alongside the NA
        (
            [np.array([1.0, 2.0]), 4, None],
            np.array([2.5, 3.0]),
            np.array([np.nan, np.nan]),
        ),
        (
            [np.array([1.0, 2.0]), Fraction(1, 2), np.nan],
            np.array([0.75, 1.25]),
            np.array([np.nan, np.nan]),
        ),
        # complex results propagate a complex NaN
        (
            [np.array([1.0, 2.0]), 1j, None],
            np.array([0.5 + 0.5j, 1 + 0.5j]),
            np.array([complex(np.nan, np.nan)] * 2),
        ),
        # integer elements with an NA
        (
            [np.array([200, 100], dtype="uint8"), pd.NA, np.array([100, 250], "uint8")],
            np.array([150.0, 175.0]),
            np.array([np.nan, np.nan]),
        ),
    ],
)
def test_object_mean_of_ndarrays_skipna_mixed(
    data, expected_skipna, expected_no_skipna
):
    ser = _object_series(data)
    tm.assert_numpy_array_equal(ser.mean(skipna=True), expected_skipna)
    tm.assert_numpy_array_equal(ser.mean(skipna=False), expected_no_skipna)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize(
    "data, match",
    [
        ([np.array([1.0, 2.0]), "x", None], "'x'"),
        ([np.array([1.0, 2.0]), None, {}], r"\{\}"),
        ([np.array([1.0, 2.0]), Decimal(1), np.nan], r"Decimal\('1'\)"),
        ([np.array([1.0, 2.0]), pd.NA, "3"], "'3'"),
    ],
)
def test_object_mean_of_ndarrays_skipna_does_not_hide_invalid(data, match, skipna):
    # an NA alongside invalid data does not turn the error into a NaN result
    ser = _object_series(data)
    with pytest.raises(TypeError, match=f"Could not convert {match} to numeric"):
        ser.mean(skipna=skipna)

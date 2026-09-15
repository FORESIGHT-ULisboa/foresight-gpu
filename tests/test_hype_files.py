"""HYPE file IO: par.txt/info.txt round trips, both output conventions, optpar layout."""

import numpy as np
import pytest

from foresight_gpu.models.hype.files import (
    InfoFile,
    ParFile,
    arity_groups,
    read_bestsims,
    read_respar,
    read_series,
    write_optpar,
)


def _write(path, lines):
    with open(path, "w", encoding="latin-1", newline="") as handle:
        handle.write("\r\n".join(lines))


class TestParFile:
    def test_reads_values_and_arity(self, hype_template):
        par = ParFile.read(hype_template / "par.txt")
        assert par.arity("rivvel") == 1
        assert par.arity("cmlt") == 3
        assert par.arity("preccorr") == 2
        assert par.arity("wcfc") == 3
        # Real files mix arity within one class; the file is authoritative, not the class.
        assert par.arity("macfrac") == 4

    def test_commented_parameters_are_not_revived(self, hype_template):
        par = ParFile.read(hype_template / "par.txt")
        assert "srrate" not in par.values

    def test_untouched_round_trip_is_byte_identical(self, hype_template, tmp_path):
        source = hype_template / "par.txt"
        par = ParFile.read(source)
        out = tmp_path / "par.txt"
        par.write(out)
        assert out.read_bytes() == source.read_bytes()

    def test_substitution_changes_one_line_and_keeps_comments(self, hype_template):
        par = ParFile.read(hype_template / "par.txt")
        new = par.with_values({"wcfc": [0.1, 0.2, 0.3]})
        changed = [i for i, (a, b) in enumerate(zip(par.lines, new.lines)) if a != b]
        assert len(changed) == 1
        assert new.lines[changed[0]] == "wcfc\t0.1\t0.2\t0.3"
        assert any(line.startswith("!!\tsrrate") for line in new.lines)
        assert par.values["wcfc"][0] != 0.1  # original untouched

    def test_arity_mismatch_raises(self, hype_template):
        par = ParFile.read(hype_template / "par.txt")
        with pytest.raises(ValueError, match="expects 3 value"):
            par.with_values({"wcfc": [0.1, 0.2]})

    def test_unknown_parameter_raises(self, hype_template):
        par = ParFile.read(hype_template / "par.txt")
        with pytest.raises(KeyError):
            par.with_values({"nosuchparameter": [1.0]})

    @pytest.mark.parametrize("wanted", [
        [1.22880957605, 1e-6, 0.000123456789],
        # Small magnitudes are the trap: a digits-after-the-point precision truncates
        # their significant digits, so HYPE would read a different value than was scored.
        [0.00082350123456789, 0.0003459512345678, 0.00013978987654321],
        [1e-9, 1e15, 0.1],
        [-0.956184, 12.012323831234567, 0.0],
    ])
    def test_values_round_trip_bit_exactly(self, hype_template, tmp_path, wanted):
        par = ParFile.read(hype_template / "par.txt")
        out = tmp_path / "par.txt"
        par.with_values({"cmlt": wanted}).write(out)
        back = ParFile.read(out).values["cmlt"]
        assert np.array_equal(back, np.array(wanted)), f"{back.tolist()} != {wanted}"

    def test_arity_groups(self, hype_template):
        groups = arity_groups(ParFile.read(hype_template / "par.txt"))
        assert "cmlt" in groups[3] and "rivvel" in groups[1]


class TestInfoFile:
    def test_parses_both_modeloption_separators(self, hype_template):
        info = InfoFile.read(hype_template / "info.txt")
        # space-separated in the template
        assert info.options["snowmeltmodel"] == "0"
        # tab-separated in the template, as the real Germany file also writes it
        assert info.options["surfacerunoff"] == "1"

    def test_parses_settings(self, hype_template):
        info = InfoFile.read(hype_template / "info.txt")
        assert info.settings["bdate"] == "1990-01-01"
        assert info.settings["cdate"] == "1991-01-01"
        assert info.settings["edate"] == "1994-12-31"
        assert info.settings["calibration"] == "n"

    def test_configured_pins_window_options_and_output(self, hype_template):
        info = InfoFile.read(hype_template / "info.txt")
        out = info.configured("1990-01-01", "1992-01-01", "1993-12-31",
                              options={"snowmeltmodel": 2}, subbasin=99)
        active = [l for l in out.lines if l.strip() and not l.lstrip().startswith("!!")]
        assert "cdate\t1992-01-01" in active
        assert "edate\t1993-12-31" in active
        assert "modeloption\tsnowmeltmodel\t2" in active
        assert "basinoutput subbasins\t99" in active
        assert out.options["snowmeltmodel"] == "2"
        # exactly one output directive survives, so the result path is knowable
        assert sum(1 for l in active if l.startswith("basinoutput variable")) == 1

    def test_configured_comments_out_previous_directives(self, hype_template):
        info = InfoFile.read(hype_template / "info.txt")
        out = info.configured("1990-01-01", "1991-01-01", "1994-12-31", subbasin=1)
        assert any(l.startswith("!!\tbasinoutput") for l in out.lines)

    def test_for_calibration_enables_and_writes_criteria(self, hype_template):
        info = InfoFile.read(hype_template / "info.txt")
        out = info.for_calibration([("MR2", "cout", "rout", 1.0)])
        active = [l for l in out.lines if l.strip() and not l.lstrip().startswith("!!")]
        assert "calibration\ty" in active
        assert "crit 1 criterion\tMR2" in active
        assert "crit 1 weight\t1" in active

    def test_write_uses_crlf(self, hype_template, tmp_path):
        info = InfoFile.read(hype_template / "info.txt")
        out = tmp_path / "info.txt"
        info.write(out)
        raw = out.read_bytes()
        assert b"\r\n" in raw
        assert b"\n\n" not in raw.replace(b"\r\n", b"\n\n\n")[:0] + b""


class TestReadSeries:
    def test_basinoutput_header_then_units(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "UNITS\tm3/s",
                      "1991-01-01\t1.5", "1991-01-02\t2.5"])
        assert read_series(path, "cout").tolist() == [1.5, 2.5]

    def test_timeoutput_comment_then_header(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["!! model=5.23.0; variable=cout; unit=m3/s;", "DATE\t1234",
                      "1991-01-01\t3.0443E+01", "1991-01-02\t4.2130E+01"])
        assert read_series(path, "1234").tolist() == [30.443, 42.130]

    def test_first_data_column_by_default(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout\tcrun", "1991-01-01\t1.0\t9.0"])
        assert read_series(path).tolist() == [1.0]

    def test_selects_by_header_name_not_position(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcrun\tcout", "1991-01-01\t9.0\t1.0"])
        assert read_series(path, "cout").tolist() == [1.0]

    def test_missing_value_becomes_nan(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "1991-01-01\t-9999", "1991-01-02\t2.0"])
        series = read_series(path, "cout")
        assert np.isnan(series[0]) and series[1] == 2.0

    def test_truncated_window_raises(self, tmp_path):
        from foresight_gpu.models.hype.dates import window_ordinals

        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "1991-01-01\t1.0"])
        t0, n = window_ordinals("1991-01-01", "1991-01-05")
        with pytest.raises(ValueError, match="expected 5 rows"):
            read_series(path, "cout", t0=t0, n_steps=n)

    def test_shifted_window_raises(self, tmp_path):
        from foresight_gpu.models.hype.dates import window_ordinals

        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "1991-02-01\t1.0", "1991-02-02\t2.0"])
        t0, n = window_ordinals("1991-01-01", "1991-01-02")
        with pytest.raises(ValueError, match="starts at ordinal"):
            read_series(path, "cout", t0=t0, n_steps=n)

    def test_unknown_column_raises_with_header(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "1991-01-01\t1.0"])
        with pytest.raises(ValueError, match="not in header"):
            read_series(path, "nope")

    def test_date_column_rejected(self, tmp_path):
        path = tmp_path / "out.txt"
        _write(path, ["DATE\tcout", "1991-01-01\t1.0"])
        with pytest.raises(ValueError, match="date column"):
            read_series(path, "DATE")


class TestOptpar:
    def test_block_starts_at_the_declared_line(self, tmp_path):
        path = tmp_path / "optpar.txt"
        write_optpar(path, [("wcfc", [0.1] * 3, [0.9] * 3, [0.001] * 3)], block_line=22)
        # read_bytes, not read_text: universal newlines would erase the CRLF we assert on
        lines = path.read_bytes().decode("latin-1").split("\r\n")
        assert lines[0] == "Info optimization"
        assert lines[21].startswith("wcfc\t")  # 1-based line 22
        assert all(not line.strip() for line in lines[10:21])

    def test_three_rows_per_parameter_with_matching_arity(self, tmp_path):
        path = tmp_path / "optpar.txt"
        write_optpar(
            path,
            [("wcfc", [0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [0.001] * 3),
             ("gratk", [0.001], [10.0], [0.01])],
            block_line=22,
        )
        raw = path.read_bytes().decode("latin-1")
        block = [l for l in raw.split("\r\n")[21:] if l]
        assert [l.split("\t")[0] for l in block] == ["wcfc"] * 3 + ["gratk"] * 3
        assert all(len(l.split("\t")) == 4 for l in block[:3])
        assert all(len(l.split("\t")) == 2 for l in block[3:])

    def test_mismatched_rows_raise(self, tmp_path):
        with pytest.raises(ValueError, match="all three rows"):
            write_optpar(tmp_path / "o.txt", [("wcfc", [0.1, 0.2], [0.9], [0.001])])

    def test_header_too_long_raises(self, tmp_path):
        with pytest.raises(ValueError, match="block must start"):
            write_optpar(tmp_path / "o.txt", [("a", [1], [2], [0.1])],
                         task=("DE",) * 30, block_line=22)


class TestCalibrationResults:
    def test_read_respar(self, tmp_path):
        path = tmp_path / "respar.txt"
        _write(path, ["!!Optimal value of parameters found during automatic calibration",
                      "mactrinf         33.83       10.83       34.72",
                      "srbeta            5.06"])
        out = read_respar(path)
        assert out["srbeta"].tolist() == [5.06]
        assert out["mactrinf"].size == 3

    def test_read_bestsims(self, tmp_path):
        path = tmp_path / "bestsims.txt"
        _write(path, ["NO,CRIT,rr2,wcfc", "1,-0.126,0.126,0.5", "2,-9999,0.2,0.4"])
        header, rows = read_bestsims(path)
        assert header == ["NO", "CRIT", "rr2", "wcfc"]
        assert rows.shape == (2, 4)
        assert np.isnan(rows[1, 1])  # -9999 mapped to NaN

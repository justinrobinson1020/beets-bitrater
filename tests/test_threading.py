"""Tests for consolidated thread-clamping module."""

import os


class TestThreadVars:
    """Tests for THREAD_VARS constant."""

    def test_thread_vars_has_all_nine_env_vars(self) -> None:
        """THREAD_VARS should contain all 9 thread-limiting env vars."""
        from bitrater._threading import THREAD_VARS

        expected_vars = {
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "KMP_BLOCKTIME",
        }
        assert set(THREAD_VARS.keys()) == expected_vars

    def test_thread_vars_values_are_strings(self) -> None:
        """All THREAD_VARS values should be strings."""
        from bitrater._threading import THREAD_VARS

        for key, value in THREAD_VARS.items():
            assert isinstance(value, str), f"{key} value should be str, got {type(value)}"


class TestClampThreads:
    """Tests for clamp_threads() - uses setdefault (for package init)."""

    def test_clamp_threads_sets_all_vars(self, monkeypatch) -> None:
        """clamp_threads should set all 9 env vars via setdefault."""
        from bitrater._threading import THREAD_VARS

        # Clear all thread vars
        for var in THREAD_VARS:
            monkeypatch.delenv(var, raising=False)

        from bitrater._threading import clamp_threads

        clamp_threads()

        for var, expected in THREAD_VARS.items():
            assert os.environ.get(var) == expected, f"{var} should be {expected}"

    def test_clamp_threads_preserves_existing_values(self, monkeypatch) -> None:
        """clamp_threads should not override existing env vars (uses setdefault)."""
        from bitrater._threading import clamp_threads

        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setenv("MKL_NUM_THREADS", "8")

        clamp_threads()

        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "8"


class TestClampThreadsHard:
    """Tests for clamp_threads_hard() - uses direct assignment (for workers)."""

    def test_clamp_threads_hard_sets_all_vars(self, monkeypatch) -> None:
        """clamp_threads_hard should set all 9 env vars."""
        from bitrater._threading import THREAD_VARS, clamp_threads_hard

        # Clear all thread vars
        for var in THREAD_VARS:
            monkeypatch.delenv(var, raising=False)

        clamp_threads_hard()

        for var, expected in THREAD_VARS.items():
            assert os.environ.get(var) == expected, f"{var} should be {expected}"

    def test_clamp_threads_hard_overrides_existing_values(self, monkeypatch) -> None:
        """clamp_threads_hard should override existing env vars (hard clamp)."""
        from bitrater._threading import clamp_threads_hard

        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setenv("MKL_NUM_THREADS", "8")

        clamp_threads_hard()

        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"

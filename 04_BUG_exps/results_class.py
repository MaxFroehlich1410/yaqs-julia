"""
This file implements a class to hold simulation results.
"""
from __future__ import annotations
from typing import Self, TYPE_CHECKING
import re
from copy import deepcopy, copy

import numpy as np
from h5py import File

from .ising_sim_util import BUGKind, TIMES_KEY

if TYPE_CHECKING:
    import numpy.typing as npt

class SimResults:
    """Class to hold test results for comparison."""
    def __init__(self):
        """
        Initialises an empty SimResults object.
        """
        self.times: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.adaptive_results: list[npt.NDArray[np.float64]] = []
        self.fixed_results: list[npt.NDArray[np.float64]] = []
        self.double_adaptive_results: list[npt.NDArray[np.float64]] = []
        self.double_fixed_results: list[npt.NDArray[np.float64]] = []
        self.hybrid_results: list[npt.NDArray[np.float64]] = []
        self.single_site_tdvp_results: list[npt.NDArray[np.float64]] = []
        self.two_site_tdvp_results: list[npt.NDArray[np.float64]] = []
        self.qutip_results: list[npt.NDArray[np.float64]] = []
        self.metadata: dict[str, str] = {}

    def load_qutip_results(self,
                           file: str | File) -> None:
        """Load qutip results from HDF5 file.

        Args:
            file: HDF5 file path or opened File object.
        """
        if isinstance(file, str):
            with File(file, 'r') as f:
                self.load_qutip_results(f)
        elif "qutip" in file.keys():
            self.load_qutip_results(file["qutip"])
        else:
            res = {}
            for key in file.keys():
                if re.match(r"\d+$", key):
                    site = int(key)
                    res[site] = file[key][:]
                if key == TIMES_KEY:
                    if self.times.size == 0:
                        self.times = file[key][:]
                    else:
                        if not np.allclose(self.times, file[key][:]):
                            errstr = "Time arrays do not match between BUG and qutip results!"
                            raise ValueError(errstr)
            self.qutip_results = [res[site]
                                  for site in sorted(res.keys())]

    def obtain_res_by_kind(self,
                           bug_kind: BUGKind
                           ) -> list[np.ndarray]:
        """Get results corresponding to the specified BUG kind.

        Args:
            bug_kind: The BUG kind.

        Returns:
            Corresponding results as a list of numpy arrays.
        """
        if bug_kind == BUGKind.FIXED:
            return self.fixed_results
        if bug_kind == BUGKind.ADAPTIVE:
            return self.adaptive_results
        if bug_kind == BUGKind.DOUBLEFIXED:
            return self.double_fixed_results
        if bug_kind == BUGKind.DOUBLEADAPTIVE:
            return self.double_adaptive_results
        if bug_kind == BUGKind.HYBRID:
            return self.hybrid_results
        if bug_kind == BUGKind.SINGLE_SITE_TDVP:
            return self.single_site_tdvp_results
        if bug_kind == BUGKind.TWO_SITE_TDVP:
            return self.two_site_tdvp_results
        raise ValueError(f"Unknown BUG kind: {bug_kind}")

    def set_res_by_kind(self,
                        bug_kind: BUGKind,
                        results: list[np.ndarray]
                        ) -> None:
        """Set results corresponding to the specified BUG kind.
        
        Args:
            bug_kind: The BUG kind.
            results: Results to set as a list of numpy arrays.
        """
        if bug_kind == BUGKind.FIXED:
            self.fixed_results = results
        elif bug_kind == BUGKind.ADAPTIVE:
            self.adaptive_results = results
        elif bug_kind == BUGKind.DOUBLEFIXED:
            self.double_fixed_results = results
        elif bug_kind == BUGKind.DOUBLEADAPTIVE:
            self.double_adaptive_results = results
        elif bug_kind == BUGKind.HYBRID:
            self.hybrid_results = results
        elif bug_kind == BUGKind.SINGLE_SITE_TDVP:
            self.single_site_tdvp_results = results
        elif bug_kind == BUGKind.TWO_SITE_TDVP:
            self.two_site_tdvp_results = results
        else:
            raise ValueError(f"Unknown BUG kind: {bug_kind}!")

    def compute_error(self,
                      bug_kind: BUGKind,
                      absolute: bool = True
                      ) -> list[np.ndarray]:
        """Compute error between BUG results and qutip results.

        Args:
            bug_kind: The BUG kind.
            absolute: Whether to compute absolute value of the error.

        Returns:
            List of error arrays for each site.
        """
        bug_results = self.obtain_res_by_kind(bug_kind)
        if not self.qutip_results:
            raise ValueError("qutip results not loaded!")
        errors = []
        for i, site in enumerate(bug_results):
            if len(site) != len(self.qutip_results[i]):
                errstr = f"Length mismatch for site {i}: BUG has {len(site)} points, qutip has {len(self.qutip_results[i])} points!"
                raise ValueError(errstr)
            if absolute:
                errors.append(np.abs(site - self.qutip_results[i]))
            else:
                errors.append(site - self.qutip_results[i])
        return errors

    def save_to_file(self,
                     file: str | File) -> None:
        """Save results to HDF5 file.

        Args:
            file: HDF5 file path or opened File object.
        """
        if isinstance(file, str):
            with File(file, 'w') as f:
                self.save_to_file(f)
        else:
            file.create_dataset(TIMES_KEY, data=self.times)
            for kind in BUGKind:
                res = self.obtain_res_by_kind(kind)
                grp = file.create_group(kind.file_key())
                for site_idx, site_data in enumerate(res):
                    grp.create_dataset(str(site_idx), data=site_data)
            for site_idx, site_data in enumerate(self.qutip_results):
                grp = file.require_group("qutip")
                grp.create_dataset(str(site_idx), data=site_data)
            for key, value in self.metadata.items():
                file.attrs[key] = value
            file.flush()

    def load_bug_res_from_file(self,
                               bug_kind: BUGKind,
                               file: str | File) -> None:
        """Load BUG results of specified kind from HDF5 file.

        Args:
            bug_kind: The BUG kind.
            file: HDF5 file path or opened File object.
        """
        if isinstance(file, str):
            with File(file, 'r') as f:
                self.load_bug_res_from_file(bug_kind, f)
        else:
            res = {}
            kind_key = bug_kind.file_key()
            for key in file.keys():
                if key == kind_key:
                    grp = file[key]
                    for site_key in grp.keys():
                        res[int(site_key)] = grp[site_key][:]
            bug_res = [res[site] for site in sorted(res.keys())]
            self.set_res_by_kind(bug_kind, bug_res)

    def load_from_file(self,
                       file: str | File) -> None:
        """Load all results from HDF5 file.

        Args:
            file: HDF5 file path or opened File object.
        """
        if isinstance(file, str):
            with File(file, 'r') as f:
                self.load_from_file(f)
        else:
            for kind in BUGKind:
                self.load_bug_res_from_file(kind, file)
            self.load_qutip_results(file)
            if TIMES_KEY in file.keys():
                self.times = file[TIMES_KEY][:]
            for key, value in file.attrs.items():
                self.metadata[key] = value

    def final_time(self) -> float:
        """Get final time of the simulation.

        Returns:
            Final time value.
        """
        if self.times.size == 0:
            raise ValueError("Time array is empty!")
        return self.times[-1]

    def num_time_steps(self) -> int:
        """Get number of time steps in the simulation.

        Returns:
            Number of time steps.
        """
        return self.times.size

    def has_bug_results(self,
                        bug_kind: BUGKind) -> bool:
        """Check if BUG results of specified kind are loaded.

        Args:
            bug_kind: The BUG kind.
        
        Returns:
            True if results of the specified BUG kind are available, False otherwise.
        """
        res = self.obtain_res_by_kind(bug_kind)
        return len(res) > 0

    def has_qutip_results(self) -> bool:
        """Check if qutip results are loaded.

        Returns:
            True if qutip results are available, False otherwise.
        """
        return len(self.qutip_results) > 0

    def make_error_results(self,
                           absolute: bool = True
                           ) -> Self:
        """Create a new SimResults object containing error results.

        Args:
            absolute: Whether to compute absolute value of the error.
        
        Returns:
            New SimResults object with error results.
        """
        if not self.has_qutip_results():
            raise ValueError("qutip results not loaded!")
        error_results = self.__class__()
        error_results.times = deepcopy(self.times)
        for bug_kind in BUGKind:
            if self.has_bug_results(bug_kind):
                error = self.compute_error(bug_kind, absolute=absolute)
                error_results.set_res_by_kind(bug_kind, error)
        error_results.metadata = copy(self.metadata)
        return error_results

    def full_rms(self) -> dict[BUGKind, float]:
        """Compute full RMS error for each BUG kind.

        Returns:
            Dictionary mapping BUGKind to full RMS error.
        """
        if not self.has_qutip_results():
            raise ValueError("qutip results not loaded!")
        rms_errors = {}
        for bug_kind in BUGKind:
            if self.has_bug_results(bug_kind):
                error = self.compute_error(bug_kind, absolute=True)
                total_squared_error = 0.0
                total_points = 0
                for site_error in error:
                    total_squared_error += np.sum(site_error**2)
                    total_points += site_error.size
                rms_errors[bug_kind] = np.sqrt(total_squared_error / total_points)
        return rms_errors

    def minimum_error(self,
                      exclude_first: int = 0
                      ) -> dict[BUGKind, float]:
        """Compute minimum error over time for each BUG kind.

        Args:
            exclude_first: Number of initial time points to exclude from consideration.

        Returns:
            Dictionary mapping BUGKind to minimum error.
        """
        if not self.has_qutip_results():
            raise ValueError("qutip results not loaded!")
        min_errors = {}
        for bug_kind in BUGKind:
            if self.has_bug_results(bug_kind):
                error = self.compute_error(bug_kind, absolute=True)
                total_error = np.zeros(self.times.shape, dtype=np.float64)
                for site_error in error:
                    total_error += site_error
                min_errors[bug_kind] = np.min(total_error[exclude_first:])
        return min_errors

    def maximum_error(self) -> dict[BUGKind, float]:
        """Compute maximum error over time for each BUG kind.

        Returns:
            Dictionary mapping BUGKind to maximum error.
        """
        if not self.has_qutip_results():
            raise ValueError("qutip results not loaded!")
        max_errors = {}
        for bug_kind in BUGKind:
            if self.has_bug_results(bug_kind):
                error = self.compute_error(bug_kind, absolute=True)
                total_error = np.zeros(self.times.shape, dtype=np.float64)
                for site_error in error:
                    total_error += site_error
                max_errors[bug_kind] = np.max(total_error)
        return max_errors

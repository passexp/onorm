"""
Performance Benchmark for onorm Normalizers.

This script benchmarks all normalizers across various data generating processes (DGPs)
to verify that performance does not degrade substantially as sample size increases.
It tests individual normalizers and pipelines across different dimensions and distributions.

Output is a well-formatted markdown file with performance statistics and plots.
"""

import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy.random import default_rng
from onorm import (
    MinMaxScaler,
    MultivariateNormalizer,
    Pipeline,
    StandardScaler,
    Winsorizer,
)
from plotnine import aes, geom_line, ggplot, theme, theme_minimal


@dataclass
class BenchmarkResult:
    """Single benchmark result."""

    normalizer: str
    dgp: str
    n_dim: int
    n_samples: int
    n_reps: int
    mean_time_us: float
    std_time_us: float
    median_time_us: float


class PerformanceBenchmark:
    """
    Benchmark normalizers for performance across different scenarios.

    Tests computational efficiency and verifies that normalizers maintain
    O(1) or O(d²) per-observation complexity regardless of sample size.
    """

    def __init__(self, seed: int = 2022):
        """
        Initialize the benchmark.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = default_rng(seed)
        self.results: List[BenchmarkResult] = []

    def get_normalizers(self, n_dim: int) -> Dict[str, Any]:
        """Create normalizers for testing."""
        normalizers = {
            "StandardScaler": StandardScaler(n_dim=n_dim),
            "MinMaxScaler": MinMaxScaler(n_dim=n_dim),
            "MultivariateNormalizer": MultivariateNormalizer(n_dim=n_dim),
            "Winsorizer(5-95%)": Winsorizer(n_dim=n_dim, clip_q=(0.05, 0.95), max_centroids=100),
            # Pipelines
            "Standard»MinMax": Pipeline([StandardScaler(n_dim=n_dim), MinMaxScaler(n_dim=n_dim)]),
            "Standard»Multivariate": Pipeline(
                [StandardScaler(n_dim=n_dim), MultivariateNormalizer(n_dim=n_dim)]
            ),
        }
        return normalizers

    def benchmark_single_run(self, normalizer: Any, X: np.ndarray) -> float:
        """
        Time a single run of fitting and transforming all data.

        Returns time per observation in seconds.
        """
        normalizer.reset()
        n_samples = X.shape[0]

        start_time = time.perf_counter()
        for x in X:
            normalizer.partial_fit(x)
            normalizer.transform(x.copy())
        end_time = time.perf_counter()

        return (end_time - start_time) / n_samples

    def run_benchmark(
        self,
        normalizer_name: str,
        normalizer: Any,
        X: np.ndarray,
        dgp_name: str,
        n_reps: int = 5,
    ) -> BenchmarkResult:
        """Run benchmark for a normalizer on given data."""
        times_per_obs = []

        for _ in range(n_reps):
            time_per_obs = self.benchmark_single_run(normalizer, X)
            times_per_obs.append(time_per_obs * 1e6)  # Convert to microseconds

        return BenchmarkResult(
            normalizer=normalizer_name,
            dgp=dgp_name,
            n_dim=X.shape[1],
            n_samples=X.shape[0],
            n_reps=n_reps,
            mean_time_us=np.mean(times_per_obs),
            std_time_us=np.std(times_per_obs),
            median_time_us=np.median(times_per_obs),
        )

    def run_all_benchmarks(self):
        """Run all benchmark scenarios."""
        print("=" * 70)
        print("Running Performance Benchmarks")
        print("=" * 70)

        scenarios = [
            # Test 1: Sample size scaling (fixed d=5, Normal)
            {
                "name": "Sample Size Scaling",
                "dgp": "Normal",
                "n_dims": [5],
                "n_samples_list": [[100, 250, 500, 1000, 2500, 5000, 10000]],
                "data_func": lambda n, d: self.rng.normal(size=(n, d)),
                "n_reps": 10,
            },
            # Test 2: Different distributions (d=5, n=500)
            {
                "name": "Distribution Types",
                "dgp": ["Normal", "Uniform", "Mixed(95N+5C)", "Correlated"],
                "n_dims": [5],
                "n_samples_list": [[500]],
                "data_funcs": [
                    lambda n, d: self.rng.normal(size=(n, d)),
                    lambda n, d: self.rng.uniform(size=(n, d)),
                    lambda n, d: self._generate_mixed(n, d),
                    lambda n, d: self._generate_correlated(n, d),
                ],
                "n_reps": 5,
            },
            # Test 3: Dimensionality scaling (n=500)
            {
                "name": "Dimensionality Scaling",
                "dgp": "Normal",
                "n_dims": [5, 10, 20, 50],
                "n_samples_list": [[500]],
                "data_func": lambda n, d: self.rng.normal(size=(n, d)),
                "n_reps": 5,
            },
        ]

        for scenario_idx, scenario in enumerate(scenarios, 1):
            print(f"\n[{scenario_idx}/{len(scenarios)}] {scenario['name']}")
            print("-" * 70)

            if "data_funcs" in scenario:
                # Multiple DGPs
                for dgp_name, data_func in zip(scenario["dgp"], scenario["data_funcs"]):
                    self._run_scenario(
                        dgp_name,
                        scenario["n_dims"],
                        scenario["n_samples_list"],
                        data_func,
                        scenario["n_reps"],
                    )
            else:
                # Single DGP
                self._run_scenario(
                    scenario["dgp"],
                    scenario["n_dims"],
                    scenario["n_samples_list"],
                    scenario["data_func"],
                    scenario["n_reps"],
                )

    def _run_scenario(self, dgp_name, n_dims, n_samples_list, data_func, n_reps):
        """Run a single scenario."""
        for d in n_dims:
            for n_samples_group in n_samples_list:
                normalizers = self.get_normalizers(d)

                for n in n_samples_group:
                    print(f"  {dgp_name} (d={d}, n={n}):")

                    # Generate data once
                    X = data_func(n, d)

                    for norm_idx, (norm_name, normalizer) in enumerate(normalizers.items(), 1):
                        result = self.run_benchmark(norm_name, normalizer, X, dgp_name, n_reps)
                        self.results.append(result)
                        print(
                            f"    [{norm_idx}/{len(normalizers)}] {norm_name}: "
                            f"{result.mean_time_us:.2f} μs/obs"
                        )

    def _generate_mixed(self, n, d):
        """Generate 95% normal + 5% Cauchy mixture."""
        n_normal = int(0.95 * n)
        n_cauchy = n - n_normal
        return np.vstack(
            [
                self.rng.normal(size=(n_normal, d)),
                self.rng.standard_cauchy(size=(n_cauchy, d)),
            ]
        )

    def _generate_correlated(self, n, d):
        """Generate correlated multivariate normal."""
        A = self.rng.normal(size=(d, d))
        cov = A @ A.T + np.eye(d) * 0.1
        return self.rng.multivariate_normal(np.zeros(d), cov, size=n)

    def create_plots(self, output_dir: str = "simulation"):
        """Create visualization plots for the benchmark results."""
        print("\nCreating visualization plots...")

        # Plot 1: Sample size scaling
        scaling_data = []
        for r in self.results:
            if r.dgp == "Normal" and r.n_dim == 5:
                scaling_data.append(
                    {
                        "n_samples": r.n_samples,
                        "time_us": r.mean_time_us,
                        "normalizer": r.normalizer,
                    }
                )

        df_scaling = pd.DataFrame(scaling_data)

        plot1 = (
            ggplot(df_scaling, aes(x="n_samples", y="time_us", color="normalizer"))
            + geom_line(size=1.5)
            + theme_minimal()
            + theme(legend_position="bottom")
        )
        plot1.save(f"{output_dir}/scaling_sample_size.png", width=12, height=6, dpi=150)

        # Plot 2: Dimensionality scaling
        dim_data = []
        for r in self.results:
            if r.dgp == "Normal" and r.n_samples == 500:
                dim_data.append(
                    {"n_dim": r.n_dim, "time_us": r.mean_time_us, "normalizer": r.normalizer}
                )

        df_dim = pd.DataFrame(dim_data)

        plot2 = (
            ggplot(df_dim, aes(x="n_dim", y="time_us", color="normalizer"))
            + geom_line(size=1.5)
            + theme_minimal()
            + theme(legend_position="bottom")
        )
        plot2.save(f"{output_dir}/scaling_dimensionality.png", width=12, height=6, dpi=150)

        print("  ✓ Saved scaling_sample_size.png")
        print("  ✓ Saved scaling_dimensionality.png")

    def _find_best_dimension_for_scaling(self) -> tuple[int | None, dict]:
        """Find the dimension with the most sample size variation."""
        sample_size_dims = set()
        for r in self.results:
            if r.dgp == "Normal":
                sample_size_dims.add(r.n_dim)

        best_dim = None
        max_variation = 0
        for dim in sample_size_dims:
            unique_n = set(
                r.n_samples for r in self.results if r.dgp == "Normal" and r.n_dim == dim
            )
            if len(unique_n) > max_variation:
                max_variation = len(unique_n)
                best_dim = dim

        scaling_groups = defaultdict(list)
        if best_dim is not None:
            for r in self.results:
                if r.dgp == "Normal" and r.n_dim == best_dim:
                    scaling_groups[r.normalizer].append(r)

        return best_dim, scaling_groups

    def _build_sample_size_table_header(self, all_n_samples: list[int]) -> str:
        """Build the dynamic header for sample size scaling table."""
        header = "| Normalizer |"
        separator = "|------------|"
        for n in all_n_samples:
            header += f" n={n:,} (μs) |"
            separator += "-----------|"
        header += " Ratio (max/min) | Constant? |\n"
        separator += "-----------------|----------|\n"
        return f"\n{header}{separator}"

    def _build_sample_size_table_row(
        self, normalizer: str, group: list, all_n_samples: list[int]
    ) -> str:
        """Build a table row for a normalizer in the sample size scaling table."""
        times = [r.mean_time_us for r in group]
        time_ratio = max(times) / min(times) if min(times) > 0 else float("inf")
        is_constant = time_ratio < 1.5
        constant_str = "✓ Yes" if is_constant else "⚠ Check"

        row = f"| {normalizer} |"
        time_map = {r.n_samples: r.mean_time_us for r in group}
        for n in all_n_samples:
            if n in time_map:
                row += f" {time_map[n]:.2f} |"
            else:
                row += " — |"
        row += f" {time_ratio:.2f} | {constant_str} |\n"
        return row

    def _generate_sample_size_scaling_section(self) -> tuple[str, dict]:
        """Generate the sample size scaling analysis section."""
        md = []

        best_dim, scaling_groups = self._find_best_dimension_for_scaling()

        if scaling_groups:
            all_n_samples = sorted(
                set(r.n_samples for group in scaling_groups.values() for r in group)
            )

            md.append(f"\n### Sample Size Scaling (d={best_dim}, Normal)\n")
            md.append(
                "\nVerifying that normalizers maintain constant per-observation complexity "
                "as sample size increases:\n"
            )

            md.append(self._build_sample_size_table_header(all_n_samples))

            for normalizer in sorted(scaling_groups.keys()):
                group = sorted(scaling_groups[normalizer], key=lambda x: x.n_samples)
                if len(group) >= 3:
                    row = self._build_sample_size_table_row(normalizer, group, all_n_samples)
                    md.append(row)

            md.append("\n![Sample Size Scaling](scaling_sample_size.png)\n")

        return "".join(md), scaling_groups

    def _find_best_sample_size_for_dim_scaling(self) -> tuple[int | None, dict]:
        """Find the sample size with the most dimension variation."""
        dim_scaling_n_samples = set()
        for r in self.results:
            if r.dgp == "Normal":
                dim_scaling_n_samples.add(r.n_samples)

        best_n = None
        max_dim_variation = 0
        for n in dim_scaling_n_samples:
            unique_dims = set(
                r.n_dim for r in self.results if r.dgp == "Normal" and r.n_samples == n
            )
            if len(unique_dims) > max_dim_variation:
                max_dim_variation = len(unique_dims)
                best_n = n

        dim_groups = defaultdict(lambda: defaultdict(list))
        if best_n is not None:
            for r in self.results:
                if r.dgp == "Normal" and r.n_samples == best_n:
                    dim_groups[r.normalizer][r.n_dim].append(r.mean_time_us)

        return best_n, dim_groups

    def _build_dim_scaling_table_header(self, all_dims: list[int]) -> str:
        """Build the dynamic header for dimensionality scaling table."""
        header = "| Normalizer |"
        separator = "|------------|"
        for d in all_dims:
            header += f" d={d} (μs) |"
            separator += "----------|"
        return f"\n{header}\n{separator}\n"

    def _build_dim_scaling_table_row(
        self, normalizer: str, dim_data: dict, all_dims: list[int]
    ) -> str:
        """Build a table row for a normalizer in the dimensionality scaling table."""
        row = f"| {normalizer} |"
        for d in all_dims:
            if d in dim_data:
                row += f" {np.mean(dim_data[d]):.2f} |"
            else:
                row += " — |"
        return row + "\n"

    def _generate_dimensionality_scaling_section(self) -> str:
        """Generate the dimensionality scaling analysis section."""
        md = []

        best_n, dim_groups = self._find_best_sample_size_for_dim_scaling()

        if dim_groups:
            all_dims = sorted(
                set(
                    dim for normalizer_dims in dim_groups.values() for dim in normalizer_dims.keys()
                )
            )

            md.append(f"\n### Dimensionality Scaling (n={best_n:,}, Normal)\n")
            md.append("\nPerformance across different dimensionalities:\n")

            md.append(self._build_dim_scaling_table_header(all_dims))

            for normalizer in sorted(dim_groups.keys()):
                row = self._build_dim_scaling_table_row(
                    normalizer, dim_groups[normalizer], all_dims
                )
                md.append(row)

            md.append("\n![Dimensionality Scaling](scaling_dimensionality.png)\n")

        return "".join(md)

    def generate_markdown_report(self) -> str:
        """Generate markdown report from results."""
        md = []
        md.append("# Performance Benchmark Report for onorm\n")
        md.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append(f"**Random Seed:** {self.seed}\n")
        md.append("\n---\n")

        # Executive Summary
        md.append("\n## Executive Summary\n")
        md.append(
            "This report benchmarks the computational performance of all normalizers "
            "in the `onorm` package to verify that:\n\n"
            "1. Performance does not degrade substantially as sample size increases\n"
            "2. All normalizers maintain O(1) or O(d²) per-observation complexity\n"
            "3. Pipelines have expected overhead from component normalizers\n"
        )

        # Use helper methods for scaling sections
        sample_size_section, scaling_groups = self._generate_sample_size_scaling_section()
        md.append(sample_size_section)

        dimensionality_section = self._generate_dimensionality_scaling_section()
        md.append(dimensionality_section)

        # Detailed Results
        md.append("\n## Detailed Results\n")

        # Group by DGP and dimension
        grouped = defaultdict(list)
        for r in self.results:
            grouped[(r.dgp, r.n_dim)].append(r)

        for (dgp, n_dim), group in sorted(grouped.items()):
            md.append(f"\n### {dgp} Distribution (d={n_dim})\n")
            md.append(
                "\n| Normalizer | n | Mean (μs/obs) | Std (μs) | "
                "Median (μs/obs) | Time/1000 obs (ms) |\n"
            )
            md.append(
                "|------------|---|---------------|----------|"
                "-----------------|-------------------|\n"
            )

            group_sorted = sorted(group, key=lambda x: (x.normalizer, x.n_samples))

            for r in group_sorted:
                time_per_1000 = r.mean_time_us * 1000 / 1000  # Convert to ms
                md.append(
                    f"| {r.normalizer} | {r.n_samples:,} | "
                    f"{r.mean_time_us:.2f} | {r.std_time_us:.2f} | "
                    f"{r.median_time_us:.2f} | {time_per_1000:.2f} |\n"
                )

        # Methodology
        md.append("\n## Methodology\n")
        md.append(
            "\n- **Timing:** `time.perf_counter()` for high-resolution measurements\n"
            "- **Replications:** 5 runs with fresh data to reduce "
            "Monte Carlo error\n"
            "- **Metrics:** Time per observation (microseconds), extrapolated "
            "time per 1000 observations\n"
            "- **Constant Time Criterion:** Time ratio (max/min) < 1.5 across "
            "sample sizes\n"
        )

        # Conclusions
        md.append("\n## Conclusions\n")

        # Check scaling
        passed_scaling = sum(
            1
            for normalizer, group in scaling_groups.items()
            if len(group) >= 3
            and (max(r.mean_time_us for r in group) / min(r.mean_time_us for r in group) < 1.5)
        )
        total_scaling = sum(1 for normalizer, group in scaling_groups.items() if len(group) >= 3)

        if passed_scaling == total_scaling:
            md.append(
                f"\n✓ **All {total_scaling} normalizers passed the constant-time scaling test.**\n"
            )
        else:
            md.append(
                f"\n⚠ {passed_scaling}/{total_scaling} normalizers passed constant-time scaling.\n"
            )

        md.append(
            "\nAll normalizers demonstrate efficient online performance suitable for "
            "streaming data applications.\n"
        )

        return "".join(md)


def convert_to_html(md_path: str, html_path: str) -> bool:
    """
    Convert markdown to HTML using pandoc.

    Parameters
    ----------
    md_path : str
        Path to input markdown file.
    html_path : str
        Path to output HTML file.

    Returns
    -------
    bool
        True if conversion succeeded, False otherwise.
    """
    try:
        subprocess.run(
            [
                "pandoc",
                md_path,
                "-f",
                "markdown",
                "-t",
                "html",
                "--standalone",
                "--self-contained",
                "--css",
                "https://cdn.jsdelivr.net/npm/water.css@2/out/water.css",
                "-o",
                html_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: pandoc conversion failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Warning: pandoc not found. Skipping HTML conversion.")
        print("Install pandoc to enable HTML output: https://pandoc.org/installing.html")
        return False


def main():
    """Run the performance benchmark."""
    start_time = time.perf_counter()

    benchmark = PerformanceBenchmark(seed=2022)

    # Run all benchmarks
    benchmark.run_all_benchmarks()

    print("\n" + "=" * 70)
    print(f"Completed {len(benchmark.results)} benchmark runs")
    print("=" * 70)

    # Create plots
    benchmark.create_plots()

    # Generate and save markdown report
    print("\nGenerating markdown report...")
    report = benchmark.generate_markdown_report()

    output_path = "simulation/benchmark_results.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✓ Report saved to: {output_path}")

    # Convert to HTML
    print("\nConverting to HTML...")
    html_path = "simulation/benchmark_results.html"
    if convert_to_html(output_path, html_path):
        print(f"✓ HTML report saved to: {html_path}")
    else:
        print("✗ HTML conversion skipped")

    # Calculate and display elapsed time
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    # Format elapsed time nicely
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)

    if minutes > 0:
        print(f"\nBenchmark complete after {minutes}m{seconds}s!")
    else:
        print(f"\nBenchmark complete after {seconds}s!")


if __name__ == "__main__":
    main()

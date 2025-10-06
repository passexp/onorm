"""
Demonstration and benchmark of hybrid serialization for MinMaxScaler.

This example shows:
1. Serialization/deserialization with realistic data from simulator DGPs
2. Performance benchmarks for serialization speed
3. Database storage simulation
"""

import json
import time

import numpy as np

from onorm import MinMaxScaler

# Performance benchmark helper
def benchmark_serialization(scaler, n_runs=1000):
    """Benchmark serialization and deserialization speed."""
    # Warmup
    for _ in range(10):
        data = scaler.to_dict()
        MinMaxScaler.from_dict(data)

    # Benchmark serialization
    start = time.perf_counter()
    for _ in range(n_runs):
        data = scaler.to_dict()
    serialize_time = (time.perf_counter() - start) / n_runs

    # Benchmark deserialization
    start = time.perf_counter()
    for _ in range(n_runs):
        MinMaxScaler.from_dict(data)
    deserialize_time = (time.perf_counter() - start) / n_runs

    # Benchmark JSON
    start = time.perf_counter()
    for _ in range(n_runs):
        json_str = scaler.to_json()
    json_serialize_time = (time.perf_counter() - start) / n_runs

    start = time.perf_counter()
    for _ in range(n_runs):
        MinMaxScaler.from_json(json_str)
    json_deserialize_time = (time.perf_counter() - start) / n_runs

    return {
        "serialize_us": serialize_time * 1e6,
        "deserialize_us": deserialize_time * 1e6,
        "json_serialize_us": json_serialize_time * 1e6,
        "json_deserialize_us": json_deserialize_time * 1e6,
        "total_roundtrip_us": (serialize_time + deserialize_time) * 1e6,
        "json_size_bytes": len(json_str),
    }


print("=" * 70)
print("MinMaxScaler Serialization Demo & Benchmark")
print("=" * 70)

# Scenario 1: Normal distribution
print("\n" + "=" * 70)
print("Scenario 1: Training with Normal distribution")
print("=" * 70)

np.random.seed(42)
scaler_normal = MinMaxScaler(n_dim=10)

n_samples = 1000
print(f"Training on {n_samples} samples from Normal(μ=0, σ=1)...")

for _ in range(n_samples):
    x = np.random.randn(10)
    scaler_normal.partial_fit(x)

print(f"Min values (first 5): {scaler_normal.min[:5]}")
print(f"Max values (first 5): {scaler_normal.max[:5]}")

# Test transformation
x_test = np.random.randn(10)
x_norm = scaler_normal.transform(x_test.copy())
print(f"\nSample transformation:")
print(f"  Input (first 5): {x_test[:5]}")
print(f"  Output (first 5): {x_norm[:5]}")

# Serialize
data_normal = scaler_normal.to_dict()
json_str_normal = scaler_normal.to_json()

print(f"\nSerialization:")
print(f"  JSON size: {len(json_str_normal)} bytes")
print(f"  Structure keys: {list(data_normal.keys())}")

# Benchmark
print(f"\nPerformance (n_dim=10, n_samples={n_samples}):")
bench = benchmark_serialization(scaler_normal)
print(f"  Serialize:     {bench['serialize_us']:.1f} μs")
print(f"  Deserialize:   {bench['deserialize_us']:.1f} μs")
print(f"  Round-trip:    {bench['total_roundtrip_us']:.1f} μs")
print(f"  JSON encode:   {bench['json_serialize_us']:.1f} μs")
print(f"  JSON decode:   {bench['json_deserialize_us']:.1f} μs")

# Verify round-trip
restored = MinMaxScaler.from_json(json_str_normal)
x_restored = restored.transform(x_test.copy())
print(f"  ✓ Round-trip matches: {np.allclose(x_norm, x_restored)}")

# Scenario 2: Uniform distribution with higher dimensions
print("\n" + "=" * 70)
print("Scenario 2: Training with Uniform distribution (high dimensional)")
print("=" * 70)

np.random.seed(123)
scaler_uniform = MinMaxScaler(n_dim=100)

n_samples_large = 5000
print(f"Training on {n_samples_large} samples from Uniform[-5, 5]...")

for _ in range(n_samples_large):
    x = np.random.uniform(-5, 5, size=100)
    scaler_uniform.partial_fit(x)

print(f"Dimension: {scaler_uniform.n_dim}")
print(f"Min range: [{scaler_uniform.min.min():.2f}, {scaler_uniform.min.max():.2f}]")
print(f"Max range: [{scaler_uniform.max.min():.2f}, {scaler_uniform.max.max():.2f}]")

# Serialize and benchmark
json_str_large = scaler_uniform.to_json()

print(f"\nSerialization:")
print(f"  JSON size: {len(json_str_large)} bytes")

print(f"\nPerformance (n_dim=100, n_samples={n_samples_large}):")
bench_large = benchmark_serialization(scaler_uniform)
print(f"  Serialize:     {bench_large['serialize_us']:.1f} μs")
print(f"  Deserialize:   {bench_large['deserialize_us']:.1f} μs")
print(f"  Round-trip:    {bench_large['total_roundtrip_us']:.1f} μs")
print(f"  JSON encode:   {bench_large['json_serialize_us']:.1f} μs")
print(f"  JSON decode:   {bench_large['json_deserialize_us']:.1f} μs")

# Database storage simulation
print("\n" + "=" * 70)
print("Database Storage Simulation")
print("=" * 70)

# Simulate storing multiple models
models = [
    ("normal_dgp_scaler", scaler_normal),
    ("uniform_dgp_scaler", scaler_uniform),
]

db_records = []
for name, scaler in models:
    data = scaler.to_dict()
    record = {
        "id": len(db_records) + 1,
        "name": name,
        "class": data["class"],
        "n_dim": data["config"]["n_dim"],
        "config_json": json.dumps(data["config"]),
        "state_json": json.dumps(data["state"]),
        "version": data["version"],
    }
    db_records.append(record)

print(f"\nStored {len(db_records)} models in simulated database:")
print(f"{'ID':<5} {'Name':<25} {'Dims':<8} {'Config Size':<15} {'State Size':<15}")
print("-" * 70)
for rec in db_records:
    config_size = len(rec["config_json"])
    state_size = len(rec["state_json"])
    print(
        f"{rec['id']:<5} {rec['name']:<25} {rec['n_dim']:<8} "
        f"{config_size:<15} {state_size:<15}"
    )

# Simulate retrieval
print(f"\nRetrieving model '{db_records[1]['name']}' from database...")
rec = db_records[1]
restored_data = {
    "version": rec["version"],
    "class": rec["class"],
    "config": json.loads(rec["config_json"]),
    "state": json.loads(rec["state_json"]),
}
db_scaler = MinMaxScaler.from_dict(restored_data)
print(f"  ✓ Loaded {db_scaler.n_dim}-dimensional scaler")
print(f"  ✓ State verified: min shape {db_scaler.min.shape}, max shape {db_scaler.max.shape}")

# Summary statistics
print("\n" + "=" * 70)
print("Performance Summary")
print("=" * 70)

print(f"\nSmall model (n_dim=10):")
print(f"  Serialization:   ~{bench['serialize_us']:.0f} μs")
print(f"  Deserialization: ~{bench['deserialize_us']:.0f} μs")
print(f"  Storage size:    {bench['json_size_bytes']} bytes")

print(f"\nLarge model (n_dim=100):")
print(f"  Serialization:   ~{bench_large['serialize_us']:.0f} μs")
print(f"  Deserialization: ~{bench_large['deserialize_us']:.0f} μs")
print(f"  Storage size:    {bench_large['json_size_bytes']} bytes")

print(f"\nConclusion:")
print(f"  • Hybrid JSON+base64 format is fast (~{bench['total_roundtrip_us']:.0f}-"
      f"{bench_large['total_roundtrip_us']:.0f} μs round-trip)")
print(f"  • Compact storage ({bench['json_size_bytes']}-{bench_large['json_size_bytes']} bytes)")
print(f"  • Database-friendly (queryable JSON metadata)")
print(f"  • Scales well with dimensionality")

print("\n" + "=" * 70)
print("✓ Demo complete!")
print("=" * 70)

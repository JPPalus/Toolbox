import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm
from typing import Callable, Tuple, List
import numba
from numba import cuda
import cupy as cp
import scipy.signal as scpy


# Function to modify directly to test different operations on the matrix.
def apply_operation(matrix: np.ndarray) -> np.ndarray:
    """Modify this function to change the applied operation."""
    # amp = np.abs(matrix[::2] + 1j*matrix[1::2]) ** 2
    n_signaux, taille_signal = matrix.shape
    # kernel = np.random.rand(1, 30)
    kernel = np.random.rand(30)
    result = np.zeros_like(matrix)
    for i in numba.prange(n_signaux):
        result[i, :] = np.convolve(matrix[i, :], kernel, mode='same')
    return result


# If optimization with Numba is needed.
@numba.njit(parallel=True)
def apply_operation_numba(matrix: np.ndarray) -> np.ndarray:
    """Same as apply_operation but optimized with Numba."""
    # amp = np.abs(matrix[::2] + 1j*matrix[1::2]) ** 2
    n_signaux, taille_signal = matrix.shape
    # kernel = np.random.rand(1, 30)
    kernel = np.random.rand(30)
    result = np.zeros_like(matrix)
    for i in numba.prange(n_signaux):
        result[i, :] = np.convolve(matrix[i, :], kernel, mode='same')
    return result

@cuda.jit
def apply_operation_gpu(matrix, kernel, result):
    # Obtention de l'index de chaque thread pour le calcul parallèle
    i = cuda.grid(1)

    if i < matrix.shape[0]:
        result[i, :] = cp.convolve(matrix[i, :], kernel, mode='same')

def apply_operation_cuda(matrix: cp.ndarray) -> np.ndarray:
    """Same as apply_operation but optimized with Cuda."""
    matrix = cp.asarray(matrix)
    n_signaux, taille_signal = matrix.shape
    # kernel = np.random.rand(1, 30)
    kernel = cp.random.rand(30) # Génération du noyau sur le GPU
    result = cp.zeros_like(matrix) # Initialisation du résultat sur GPU

    # Définit les threads et les blocs CUDA
    # threads_per_block = 256
    # blocks_per_grid = (n_signaux + (threads_per_block - 1))

    # Appel de la fonction CUDA pour appliquer la convolution
    #apply_operation_gpu[blocks_per_grid, threads_per_block](matrix, kernel, result)

    # Appliquer la convolution sur chaque ligne avec cupy.convolve
    for i in range(n_signaux):
        result[i, :] = cp.convolve(matrix[i, :], kernel, mode='same')

    return cp.asnumpy(result)


def benchmark_gpu(
        matrix: cp.ndarray,
        operation: Callable[[cp.ndarray], cp.ndarray],
        max_rows: int,
        max_cols: int,
        num_steps: int) -> Tuple[
    List[int], List[int], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Benchmarks an operation on a given CuPy matrix by increasing row and column sizes.

    Args:
        matrix      (cp.ndarray): Initial matrix to transform.
        operation   (Callable[[cp.ndarray], cp.ndarray]): Function to apply to the matrix.
        max_rows    (int): Maximum number of rows.
        max_cols    (int): Maximum number of columns.
        num_steps   (int): Number of steps for both row and column benchmarks.

    Returns:
        Tuple[List[int], List[int], List[float], List[float], List[float], List[float], List[float], List[float]]:
            - List of tested row sizes.
            - List of tested column sizes.
            - List of execution times for increasing row sizes.
            - List of execution times for increasing column sizes.
            - List of memory usage for row sizes.
            - List of memory usage for column sizes.
            - List of CPU usage for row sizes.
            - List of CPU usage for column sizes.
    """
    row_sizes = cp.linspace(start=1, stop=max_rows, num=num_steps, dtype=int).tolist()
    col_sizes = cp.linspace(start=1, stop=max_cols, num=num_steps, dtype=int).tolist()
    row_times: List[float] = []
    col_times: List[float] = []
    row_memory_usage: List[float] = []
    col_memory_usage: List[float] = []
    row_cpu_usage: List[float] = []
    col_cpu_usage: List[float] = []

    # Benchmark by increasing row size while keeping columns fixed.
    for rows in tqdm(row_sizes, desc="Benchmarking rows", unit="iteration"):
        test_matrix: cp.ndarray = matrix[:rows, :max_cols]
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=1)
        start_time: float = time.time()
        operation(test_matrix)
        end_time: float = time.time()
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=1)

        row_times.append(end_time - start_time)
        row_memory_usage.append((mem_after - mem_before) / (1024 * 1024))  # Convert to MB
        row_cpu_usage.append(cpu_after)

    # Benchmark by increasing column size while keeping rows fixed.
    for cols in tqdm(col_sizes, desc="Benchmarking columns", unit="iteration"):
        test_matrix: cp.ndarray = matrix[:max_rows, :cols + 29]  # +29 for convolve with array (1,30)
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=1)
        start_time: float = time.time()
        operation(test_matrix)
        end_time: float = time.time()
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=1)

        col_times.append(end_time - start_time)
        col_memory_usage.append((mem_after - mem_before) / (1024 * 1024))  # Convert to MB
        col_cpu_usage.append(cpu_after)

    return row_sizes, col_sizes, row_times, col_times, row_memory_usage, col_memory_usage, row_cpu_usage, col_cpu_usage


def benchmark_cpu(
        matrix: np.ndarray,
        operation: Callable[[np.ndarray], np.ndarray],
        max_rows: int,
        max_cols: int,
        num_steps: int) -> Tuple[List[int], List[int], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Benchmarks an operation on a given NumPy matrix by increasing row and column sizes.
    
    Args:
        matrix      (np.ndarray): Initial matrix to transform.
        operation   (Callable[[np.ndarray], np.ndarray]): Function to apply to the matrix.
        max_rows    (int): Maximum number of rows.
        max_cols    (int): Maximum number of columns.
        num_steps   (int): Number of steps for both row and column benchmarks.
    
    Returns:
        Tuple[List[int], List[int], List[float], List[float], List[float], List[float], List[float], List[float]]: 
            - List of tested row sizes.
            - List of tested column sizes.
            - List of execution times for increasing row sizes.
            - List of execution times for increasing column sizes.
            - List of memory usage for row sizes.
            - List of memory usage for column sizes.
            - List of CPU usage for row sizes.
            - List of CPU usage for column sizes.
    """
    row_sizes = np.linspace(start=1, stop=max_rows, num=num_steps, dtype=int).tolist()
    col_sizes = np.linspace(start=1, stop=max_cols, num=num_steps, dtype=int).tolist()
    row_times: List[float] = []
    col_times: List[float] = []
    row_memory_usage: List[float] = []
    col_memory_usage: List[float] = []
    row_cpu_usage: List[float] = []
    col_cpu_usage: List[float] = []

    # Benchmark by increasing row size while keeping columns fixed.
    for rows in tqdm(row_sizes, desc="Benchmarking rows", unit="iteration"):
        test_matrix: np.ndarray = matrix[:rows, :max_cols]
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=1)
        start_time: float = time.time()
        operation(test_matrix)
        end_time: float = time.time()
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=1)

        row_times.append(end_time - start_time)
        row_memory_usage.append((mem_after - mem_before) / (1024 * 1024))  # Convert to MB
        row_cpu_usage.append(cpu_after)

    # Benchmark by increasing column size while keeping rows fixed.
    for cols in tqdm(col_sizes, desc="Benchmarking columns", unit="iteration"):
        test_matrix: np.ndarray = matrix[:max_rows, :cols+29]  # +29 for convolve with array (1,30)
        mem_before = psutil.virtual_memory().used
        cpu_before = psutil.cpu_percent(interval=1)
        start_time: float = time.time()
        operation(test_matrix)
        end_time: float = time.time()
        mem_after = psutil.virtual_memory().used
        cpu_after = psutil.cpu_percent(interval=1)

        col_times.append(end_time - start_time)
        col_memory_usage.append((mem_after - mem_before) / (1024 * 1024))  # Convert to MB
        col_cpu_usage.append(cpu_after)

    return row_sizes, col_sizes, row_times, col_times, row_memory_usage, col_memory_usage, row_cpu_usage, col_cpu_usage


def main() -> None:
    """
    Main function to execute the benchmark and save the plots with memory and CPU usage.
    """
    parser = argparse.ArgumentParser(description="Matrix Benchmarking")
    parser.add_argument("max_rows", type=int, help="Maximum number of rows")
    parser.add_argument("max_cols", type=int, help="Maximum number of columns")
    parser.add_argument("num_steps", type=int, help="Number of benchmarking steps")
    parser.add_argument("--iterations", type=int, default=3, help="Number of times to repeat the benchmark")
    parser.add_argument("--output", type=str, default="benchmark.png", help="Filename to save the plot")
    parser.add_argument("--use_numba", action="store_true", help="Enable Numba optimization")
    parser.add_argument("--use_cuda", action="store_true", help="Enable Cuda optimization")
    args = parser.parse_args()

    # Choose the operation function
    if args.use_cuda:
        print("cuda")
        operation = apply_operation_cuda
        benchmark = benchmark_gpu
    elif args.use_numba:
        print("numba")
        operation = apply_operation_numba
        benchmark = benchmark_cpu
    else:
        operation = apply_operation
        benchmark = benchmark_cpu

    # Prepare storage for results.
    all_row_times = []
    all_col_times = []
    all_row_memory = []
    all_col_memory = []
    all_row_cpu = []
    all_col_cpu = []

    # Run benchmark multiple times.
    for i in range(args.iterations):
        print(f"Starting iteration {i + 1}/{args.iterations} {'(Numba enabled)' if args.use_numba else ''}")
        print(f"Starting iteration {i + 1}/{args.iterations} {'(Cuda enabled)' if args.use_cuda else ''}")
        matrix: np.ndarray = np.random.rand(args.max_rows, args.max_cols)
        row_sizes, col_sizes, row_times, col_times, row_memory, col_memory, row_cpu, col_cpu = benchmark(
            matrix,
            operation,
            args.max_rows,
            args.max_cols,
            args.num_steps)
        all_row_times.append(row_times)
        all_col_times.append(col_times)
        all_row_memory.append(row_memory)
        all_col_memory.append(col_memory)
        all_row_cpu.append(row_cpu)
        all_col_cpu.append(col_cpu)

    # Compute averages.
    avg_row_times = np.mean(all_row_times, axis=0)
    avg_col_times = np.mean(all_col_times, axis=0)
    avg_row_memory = np.mean(all_row_memory, axis=0)
    avg_col_memory = np.mean(all_col_memory, axis=0)
    avg_row_cpu = np.mean(all_row_cpu, axis=0)
    avg_col_cpu = np.mean(all_col_cpu, axis=0)

    # Plot results.
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].plot(row_sizes, avg_row_times, label='Execution Time (Rows)', marker='o', color='blue')
    axes[0].plot(col_sizes, avg_col_times, label='Execution Time (Columns)', marker='o', color='orange')
    axes[0].set_title("Execution Time")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(row_sizes, avg_row_memory, label='Memory Usage (Rows)', marker='^', color='green')
    axes[1].plot(col_sizes, avg_col_memory, label='Memory Usage (Columns)', marker='^', color='brown')
    axes[1].set_title("Memory Usage")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(row_sizes, avg_row_cpu, label='CPU Usage (Rows)', marker='s', color='red')
    axes[2].plot(col_sizes, avg_col_cpu, label='CPU Usage (Columns)', marker='s', color='purple')
    axes[2].set_title("CPU Usage")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Benchmark plot saved as {args.output}")


if __name__ == "__main__":
    main()
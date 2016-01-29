/**
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.fireflysemantics.benchmark;

import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import com.fireflysemantics.math.linear.matrix.MatrixOperations;
import com.fireflysemantics.math.linear.matrix.SimpleMatrix;

@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(2)
@BenchmarkMode(Mode.AverageTime)

/**
 * For related benchmarks see: <a href=
 * "http://stackoverflow.com/questions/35037893/java-8-stream-matrix-multiplication-10x-slower-than-for-loop">
 * Java 8 Stream Matrix Multiplication 10X Slower Than For Loop?</a>
 *
 */
public class MultiplyBenchmark {

	@State(Scope.Thread)
	public static class Container {
		static Random random = new Random();

		static int HUNDRED = 100;
		static int THOUSAND = 1000;
		static int THREE_THOUSAND = 3000;
		static int TEN_K = 10000;

		static double[][] A1_HUNDRED =
				IntStream.range(0, HUNDRED).mapToObj(r -> random.doubles(HUNDRED, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] A2_THOUSAND =
				IntStream.range(0, THOUSAND).mapToObj(r -> random.doubles(THOUSAND, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] A3_THREE_THOUSAND =
				IntStream.range(0, THREE_THOUSAND)
						.mapToObj(r -> random.doubles(THREE_THOUSAND, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] A11_HUNDRED =
				IntStream.range(0, HUNDRED).mapToObj(r -> random.doubles(HUNDRED, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] A22_THOUSAND =
				IntStream.range(0, THOUSAND).mapToObj(r -> random.doubles(THOUSAND, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] A33_THREE_THOUSAND =
				IntStream.range(0, THREE_THOUSAND)
						.mapToObj(r -> random.doubles(THREE_THOUSAND, 0, 10).toArray())
						.toArray(double[][]::new);

		static double[][] TEN_K_MATRIX =
				IntStream.range(0, TEN_K).mapToObj(r -> random.doubles(TEN_K, 0, 10).toArray())
						.toArray(double[][]::new);

		public Array2DRowRealMatrix rm1_100 = new Array2DRowRealMatrix(A1_HUNDRED);
		public Array2DRowRealMatrix rm2_100 = new Array2DRowRealMatrix(A11_HUNDRED);

		public SimpleMatrix sm1_100 = new SimpleMatrix(A1_HUNDRED);
		public SimpleMatrix sm2_100 = new SimpleMatrix(A11_HUNDRED);

		public Array2DRowRealMatrix rm1_1000 = new Array2DRowRealMatrix(A22_THOUSAND);
		public Array2DRowRealMatrix rm2_1000 = new Array2DRowRealMatrix(A22_THOUSAND);

		public SimpleMatrix sm1_1000 = new SimpleMatrix(A2_THOUSAND);
		public SimpleMatrix sm2_1000 = new SimpleMatrix(A22_THOUSAND);

		public Array2DRowRealMatrix rm_10K = new Array2DRowRealMatrix(TEN_K_MATRIX);
		public SimpleMatrix sm_10K = new SimpleMatrix(TEN_K_MATRIX);
	}

	// @Benchmark
	public SimpleMatrix addScalarFM(Container c) {
		return MatrixOperations.addScalar().apply(c.sm_10K, 10d);
	}

	// @Benchmark
	public RealMatrix addScalarCM(Container c) {
		return c.rm_10K.scalarAdd(10d);
	}

	// @Benchmark
	public double traceCM(Container c) {
		return c.rm_10K.getTrace();
	}

	// @Benchmark
	public Double traceFM(Container c) {
		return MatrixOperations.trace().apply(c.sm_10K);
	}

	// @Benchmark
	public RealMatrix transposeCM(Container c) {
		return c.rm_10K.transpose();
	}

	// @Benchmark
	public SimpleMatrix transposeFM(Container c) {
		return MatrixOperations.transpose().apply(c.sm_10K);
	}

	// @Benchmark
	public double normCM(Container c) {
		return c.rm_10K.getNorm();
	}

	// @Benchmark
	public double normFM(Container c) {
		return MatrixOperations.norm().apply(c.sm_10K);
	}

	// @Benchmark
	public double frobeniusNormCM(Container c) {
		return c.rm_10K.getFrobeniusNorm();
	}

	// @Benchmark
	public double frobeniusNormFM(Container c) {
		return MatrixOperations.frobeniusNorm(false).apply(c.sm_10K);
	}

	// @Benchmark
	public RealMatrix multiplyCM100_100(Container m) {
		return m.rm1_100.multiply(m.rm2_100);
	}

	// @Benchmark
	public SimpleMatrix multiplyFM100_100(Container m) {
		return MatrixOperations.multiply().apply(m.sm1_100, m.sm2_100);
	}

	// @Benchmark
	public RealMatrix multiplyCM1000_1000(Container m) {
		return m.rm1_1000.multiply(m.rm2_1000);
	}

	// @Benchmark
	public SimpleMatrix multiplyFM1000_1000(Container m) {
		return MatrixOperations.multiply().apply(m.sm1_1000, m.sm2_1000);
	}
}

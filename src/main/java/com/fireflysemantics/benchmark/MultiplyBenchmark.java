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
import org.openjdk.jmh.annotations.Benchmark;
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

@Warmup(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 10, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(3)
@BenchmarkMode(Mode.AverageTime)

public class MultiplyBenchmark {

	@State(Scope.Thread)
	public static class Matrix {
		static Random random = new Random();

		static int HUNDRED = 100;
		static int THOUSAND = 1000;
		static int THREE_THOUSAND = 3000;

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

		public Array2DRowRealMatrix rm1_100 = new Array2DRowRealMatrix(A1_HUNDRED);
		public Array2DRowRealMatrix rm2_100 = new Array2DRowRealMatrix(A11_HUNDRED);

		public SimpleMatrix sm1_100 = new SimpleMatrix(A1_HUNDRED);
		public SimpleMatrix sm2_100 = new SimpleMatrix(A11_HUNDRED);

		public Array2DRowRealMatrix rm1_1000 = new Array2DRowRealMatrix(A22_THOUSAND);
		public Array2DRowRealMatrix rm2_1000 = new Array2DRowRealMatrix(A22_THOUSAND);

		public SimpleMatrix sm1_1000 = new SimpleMatrix(A2_THOUSAND);
		public SimpleMatrix sm2_1000 = new SimpleMatrix(A22_THOUSAND);

	}

	@Benchmark
	public RealMatrix multiplyCM100_100(Matrix m) {
		return m.rm1_100.multiply(m.rm2_100);
	}

	@Benchmark
	public SimpleMatrix multiplyFM100_100(Matrix m) {
		return MatrixOperations.multiply().apply(m.sm1_100, m.sm2_100);
	}

	@Benchmark
	public RealMatrix multiplyCM1000_1000(Matrix m) {
		return m.rm1_1000.multiply(m.rm2_1000);
	}

	@Benchmark
	public SimpleMatrix multiplyFM1000_1000(Matrix m) {
		return MatrixOperations.multiply().apply(m.sm1_1000, m.sm2_1000);
	}
}

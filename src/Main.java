import java.util.Arrays;

public class Main {
    private static final double EPS = 1e-5;

    public static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int rows = matrix.length, cols = matrix[0].length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) sum += matrix[i][j] * vector[j];
            result[i] = sum;
        }
        return result;
    }

    public static double[] addVectors(double[] vec1, double[] vec2, boolean add) {
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) result[i] = add ? vec1[i] + vec2[i] : vec1[i] - vec2[i];
        return result;
    }

    public static double[] scaleVector(double scalar, double[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) result[i] = scalar * vector[i];
        return result;
    }

    public static double[] inverseVector(double[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            if (vector[i] == 0) {
                System.out.println("The method is not applicable!");
                System.exit(1);
            }

            result[i] = 1.0 / vector[i];
        }
        return result;
    }

    public static double norm(double[] vector) {
        double sum = 0;
        for (double v : vector) sum += v * v;
        return Math.sqrt(sum);
    }

    public static boolean isFeasible(double[][] A, double[] x, double[] b) {
        double[] Ax = multiplyMatrixVector(A, x);
        for (int i = 0; i < Ax.length; i++) {
            if (Ax[i] > b[i]) return false;
        }
        return true;
    }

    public static double[] interiorPointSolve(double[][] A, double[] b, double[] c, double alphaStart, double epsilon) {
        int n = c.length;
        double[] x = new double[n];
        Arrays.fill(x, 1.0);
        int maxIters = 100000;
        double mu = 0.1;
        double alpha = alphaStart;

        for (int iter = 0; iter < maxIters; iter++) {
            double[] grad = addVectors(c, scaleVector(mu, inverseVector(x)), false);
            double[] deltaX = scaleVector(-1, grad);

            while (!isFeasible(A, addVectors(x, scaleVector(alpha, deltaX), true), b)) {
                alpha *= 0.9;
                if (alpha < epsilon) {
                    System.out.println("Alpha too small; stopping to avoid instability.");
                    return x;
                }
            }

            x = addVectors(x, scaleVector(alpha, deltaX), true);
            for (int i = 0; i < n; i++) {
                if (x[i] <= 0) x[i] = epsilon;
            }

            if (norm(deltaX) < epsilon) {
                System.out.println("Converged in " + (iter + 1) + " iterations.");
                break;
            }

            mu *= 0.95;
            alpha = alphaStart;
        }

        return x;
    }

    private static double[] simplexSolve(double[] c, double[][] a, double[] b) {
        int vars = c.length, constraints = a.length;
        double[][] tableau = new double[constraints + 1][vars + constraints + 1];

        for (int i = 0; i < constraints; i++) {
            System.arraycopy(a[i], 0, tableau[i], 0, vars);
            tableau[i][vars + i] = 1.0;
            tableau[i][vars + constraints] = b[i];
            if (b[i] < 0) return null;
        }

        for (int j = 0; j < vars; j++) tableau[constraints][j] = -c[j];

        while (true) {
            int pCol = -1;
            double minVal = 0;
            for (int j = 0; j < vars + constraints; j++) {
                if (tableau[constraints][j] < minVal - EPS) {
                    minVal = tableau[constraints][j];
                    pCol = j;
                }
            }

            if (pCol == -1) {
                double[] res = new double[vars + 1];
                for (int i = 0; i < constraints; i++) {
                    for (int j = 0; j < vars; j++) {
                        if (Math.abs(tableau[i][j] - 1.0) < EPS) res[j] = tableau[i][vars + constraints];
                    }
                }
                res[vars] = tableau[constraints][vars + constraints];
                return res;
            }

            int pRow = -1;
            double minRatio = Double.MAX_VALUE;
            for (int i = 0; i < constraints; i++) {
                if (tableau[i][pCol] > EPS) {
                    double ratio = tableau[i][vars + constraints] / tableau[i][pCol];
                    if (ratio < minRatio) {
                        minRatio = ratio;
                        pRow = i;
                    }
                }
            }

            if (pRow == -1) {
                System.out.println("Problem is unbounded");
                return null;
            }

            pivot(tableau, pRow, pCol);
        }
    }

    private static void pivot(double[][] tableau, int row, int col) {
        double pivotVal = tableau[row][col];
        for (int j = 0; j < tableau[0].length; j++) tableau[row][j] /= pivotVal;

        for (int i = 0; i < tableau.length; i++) {
            if (i != row) {
                double factor = tableau[i][col];
                for (int j = 0; j < tableau[0].length; j++) tableau[i][j] -= factor * tableau[row][j];
            }
        }
    }

    public static void testCase(double[][] A, double[] b, double[] c, double alpha, double epsilon) {
        double[] ipSolution = interiorPointSolve(A, b, c, alpha, epsilon);
        double ipValue = 0;
        for (int i = 0; i < c.length; i++) ipValue += c[i] * ipSolution[i];
        System.out.println("Interior-Point Approximate Value (Alpha=" + alpha + "): " + ipValue);

        double[] simplexSolution = simplexSolve(c, A, b);
        if (simplexSolution != null) {
            System.out.printf("Simplex Exact Value: %.5f%n", simplexSolution[simplexSolution.length - 1]);
        }
    }

    public static void runTests() {
        double epsilon = 1e-5;
        double alpha1 = 0.5;
        double alpha2 = 0.9;


        System.out.println("\nTest Case 1:");
        double[] c1 = {3, 1};
        double[][] A1 = {
                {1, 2},
                {3, 4},
                {2, 1}
        };
        double[] b1 = {4, 10, 3};
        System.out.println("input :");
        System.out.println("c=(3, 1)");
        System.out.println("""
                A = [1, 2]
                    [3, 4]
                    [2, 1]
                """);

        System.out.println("b=(4, 10, 3)");
        System.out.println("output : ");
        testCase(A1, b1, c1, alpha1, epsilon);
        testCase(A1, b1, c1, alpha2, epsilon);


        System.out.println("\nTest Case 2:");
        double[] c2 = {2, 1, 1};
        double[][] A2 = {
                {1, 1, 1},
                {2, 1, 0},
                {1, 0, 3}
        };
        double[] b2 = {5, 8, 7};
        System.out.println("c=(2, 1, 1)");
        System.out.println("""
                A = [1, 1, 1]
                    [2, 1, 0]
                    [1, 0, 3]
                """);

        System.out.println("b=(5, 8, 7)");
        testCase(A2, b2, c2, alpha1, epsilon);
        testCase(A2, b2, c2, alpha2, epsilon);


        System.out.println("\nTest Case 3:");
        double[] c4 = {5, 3, 2};
        double[][] A4 = {
                {1, 1, 1},
                {4, 2, 3},
                {2, 5, 5}
        };
        double[] b4 = {10, 20, 15};
        System.out.println("input :");
        System.out.println("c=(5, 3, 2)");
        System.out.println("""
                A = [1, 1, 1]
                    [4, 2, 3]
                    [2, 5, 5]
                """);

        System.out.println("b=(10, 20, 15)");
        System.out.println("output : ");

        testCase(A4, b4, c4, alpha1, epsilon);
        testCase(A4, b4, c4, alpha2, epsilon);


        System.out.println("\nTest Case 4:");
        double[] c5 = {3, 2};
        double[][] A5 = {
                {1, 1},
                {2, 3},
                {1, 0}
        };
        double[] b5 = {5, 12, 4};
        System.out.println("input :");
        System.out.println("c=(3, 2)");
        System.out.println("""
                A = [1, 1]
                    [2, 3]
                    [1, 0]
                """);

        System.out.println("b=(5, 12, 4)");
        System.out.println("output : ");
        testCase(A5, b5, c5, alpha1, epsilon);
        testCase(A5, b5, c5, alpha2, epsilon);

        System.out.println("\nTest Case 5:");
        double[] c = {3, 5, 4};
        double[][] A = {
                {2, 3, 0},
                {2, 5, 10},
                {3, 2, 4}
        };
        double[] b = {8, 10, 15};

        System.out.println("input :");
        System.out.println("c=(3, 5, 4)");
        System.out.println("""
                A = [2, 3, 0]
                    [2, 5, 10]
                    [3, 2, 4]
                """);

        System.out.println("b=(8, 10, 15)");
        System.out.println("output : ");
        testCase(A, b, c, alpha1, epsilon);
        testCase(A, b, c, alpha2, epsilon);


    }
    public static void main(String[] args) {
        runTests();
    }
}

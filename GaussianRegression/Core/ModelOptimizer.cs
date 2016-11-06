using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace GaussianRegression.Core
{
    class ModelOptimizer
    {
        private Matrix<double> K;
        private Matrix<double> Y;
        private List<Vector<double>> x;
        private int N;

        private readonly CovFunction cf;
        private CovMatrix covMatrix;
        
        public ModelOptimizer(CovMatrix covM, CovFunction cf)
        {
            this.cf = cf;
            this.covMatrix = covM;
            reboot();
        }

        private void reboot()
        {
            this.K = covMatrix.getK();
            this.Y = covMatrix.getY();
            this.x = covMatrix.getX();
            this.N = K.RowCount;
        }

        private static readonly double INITIALSTEP = 0.1;
        private static readonly double STEP_UP_RATIO = 1.15;
        private static readonly double STEP_DOWN_RATIO = 0.75;
        private static readonly double CONVERGENCE_THRESHOLD = 0.001;
        private static readonly double MAX_ITERATION = 200;

        public void optimize()
        {
            var typeAndHyper = cf.param;
            var previousGradient = new Dictionary<Type, double>();
            var typeAndStep = new Dictionary<Type, double>();
            var iterCounter = 0;
            var converged = false;
            //Initialize Step Sizes
            foreach (var kv in typeAndHyper)
            {
                typeAndStep.Add(kv.Key, INITIALSTEP);
            }

            while (iterCounter < MAX_ITERATION && !converged)
            {
                var currentGradient = new Dictionary<Type, double>();
                iterCounter++;
                //Compute the gradient
                foreach (var kv in typeAndHyper)
                {
                    currentGradient[kv.Key] = differentiateLogMarginal(kv.Key);
                }
                GPUtility.Log("***************************************************************");
                GPUtility.Log("Iter: " + iterCounter + " " + string.Join(", ", currentGradient.Select(kv => kv.Value).ToArray()));

                if (currentGradient.All(kv => Math.Abs(kv.Value) < CONVERGENCE_THRESHOLD))
                {
                    converged = true;
                }

                //Otherwise, compute the step sizes
                if (previousGradient.Count != 0)
                {
                    foreach (var k in typeAndStep.Keys.ToList())
                    {
                        if (Math.Sign(currentGradient[k]) == Math.Sign(previousGradient[k]))
                            typeAndStep[k] = typeAndStep[k] * STEP_UP_RATIO;
                        else
                            typeAndStep[k] = typeAndStep[k] * STEP_DOWN_RATIO;
                    }
                }
                previousGradient = currentGradient;

                //Update with new step size. Go in the direction of positive gradient
                foreach (var key in typeAndHyper.Keys.ToList())
                {
                    Hyperparam par = typeAndHyper[key];
                    //Temp Fix
                    var v = Math.Max(0.000000001, par.value + Math.Sign(currentGradient[key]) * typeAndStep[key]);
                    typeAndHyper[key] = Hyperparam.createInstance(key, v);
                }

                GPUtility.Log("Iter: " + iterCounter + " " + string.Join(", ", cf.param.Select(kv => kv.Value.value).ToArray()));
                //Update CovMatrix
                covMatrix.recalculate();
                reboot();
            }
        }

        private double evaluateLogMarginal()
        {
            var firstTerm = Y.Transpose().Multiply(K.Inverse()).Multiply(Y).ToArray()[0,0];
            var secondTerm = K.Determinant();
            var thirdTerm = N * Math.Log(2 * Math.PI) / 2;
            return -0.5 * firstTerm - 0.5 * Math.Log(secondTerm) - thirdTerm;
        }

        private double differentiateLogMarginal(Type withRespectTo)
        {
            double[,] k_partial = new double[N, N];
            for(int i = 0; i < N; i++)
            {
                for(int j = 0; j < N; j++)
                {
                    k_partial[i, j] = cf.differential(withRespectTo)(x.ElementAt(i), x.ElementAt(j));
                }
            }
            var k_inverse = K.Inverse();
            Matrix<double> K_partial = Matrix<double>.Build.DenseOfArray(k_partial);
            Matrix<double> alpha = k_inverse.Multiply(Y);
            Matrix<double> rightResult = (alpha.Multiply(alpha.Transpose()).Subtract(k_inverse)).Multiply(K_partial);
            var trace = rightResult.Trace();
            if (double.IsNaN(trace))
                throw new Exception("Invalid Result! ");
            return 0.5 * trace;
        }

    }
}

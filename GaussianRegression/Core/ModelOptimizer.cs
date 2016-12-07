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

        private List<Hyperparam> minBounds;
        private List<Hyperparam> maxBounds;

        public ModelOptimizer(CovMatrix covM, CovFunction cf, List<Hyperparam> minBounds, List<Hyperparam> maxBounds)
        {
            this.cf = cf;
            this.covMatrix = covM;
            this.minBounds = minBounds;
            this.maxBounds = maxBounds;
            reboot();
        }

        private void reboot()
        {
            covMatrix.recalculate();
            this.K = covMatrix.getK();
            this.Y = covMatrix.getY();
            this.x = covMatrix.getX();
            this.N = K.RowCount;
        }

        private static readonly double INITIALSTEP = 0.1;
        private static readonly double STEP_UP_RATIO = 1.15;
        private static readonly double STEP_DOWN_RATIO = 0.5;
        private static readonly double CONVERGENCE_THRESHOLD = 0.001;
        private static readonly double MAX_ITERATION = 200;

        public void optimize()
        {
            var typeAndHyper = cf.param;

            var minBound = new Dictionary<Type, double>();
            var maxBound = new Dictionary<Type, double>();

            var previousGradient = new Dictionary<Type, double>();
            var typeAndStep = new Dictionary<Type, double>();
            var iterCounter = 0;
            var converged = false;
            //Initialize Step Sizes
            foreach (var kv in typeAndHyper)
            {
                typeAndStep.Add(kv.Key, INITIALSTEP);
                var min = minBounds != null && minBounds.Count > 0? minBounds.Find(b => b.type.Equals(kv.Key)) : null;
                var max = maxBounds != null && maxBounds.Count > 0 ? maxBounds.Find(b => b.type.Equals(kv.Key)) : null;
                minBound.Add(kv.Key, min == null? 0 : min.value);
                maxBound.Add(kv.Key, max == null ? Double.PositiveInfinity : max.value);
            }

            while (iterCounter < MAX_ITERATION && !converged)
            {
                GPUtility.Log("*************************************************************************", GPUtility.LogLevel.DEBUG);
                GPUtility.Log("Iter: " + iterCounter + " Log Marginal Likelihood" + evaluateLogMarginal(), GPUtility.LogLevel.DEBUG);
                
                var currentGradient = new Dictionary<Type, double>();
                iterCounter++;
                //Compute the gradient
                foreach (var kv in typeAndHyper)
                {
                    currentGradient[kv.Key] = differentiateLogMarginal(kv.Key);
                }
                GPUtility.Log("Iter: " + iterCounter + " " + string.Join(", ", currentGradient.Select(kv => kv.Value).ToArray()), GPUtility.LogLevel.DEBUG);

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
                    var v = par.value + Math.Sign(currentGradient[key]) * typeAndStep[key];
                    v = Math.Max(minBound[key], v);
                    v = Math.Min(maxBound[key], v);
                    typeAndHyper[key] = Hyperparam.createInstance(key, v);
                }

                GPUtility.Log("Iter: " + iterCounter + " " + string.Join(", ", cf.param.Select(kv => kv.Value.value).ToArray()), GPUtility.LogLevel.DEBUG);
                //Update CovMatrix
                covMatrix.recalculate();
                reboot();
            }
        }

        internal void evaluateLog(Dictionary<Type, Tuple<Hyperparam, Hyperparam>> dicts)
        {
            var setOfParams = composeHyperParams(dicts);
            Dictionary<List<Hyperparam>, double> res = new Dictionary<List<Hyperparam>, double>();
            foreach (var listOfHyper in setOfParams)
            {
                //double before = cf.param.Sum(p => p.Value.value);
                cf.addParams(listOfHyper.ToArray());
                reboot();
                /*double after = cf.param.Sum(p => p.Value.value);
                if (before == after)
                    GPUtility.Log("CF Not Matched: Before: " + before + " After: " + after);*/
                res.Add(listOfHyper, evaluateLogMarginal());
            }
            FileService fs = new FileService("LogMarginal.csv");
            int counter = 0;
            int size = res.Count;
            fs.writeToFile(res.Select(
                kv => {
                    counter++;
                    GPUtility.Log(" Calculating " + counter + " Among " + size, GPUtility.LogLevel.DEBUG, true);
                    return String.Join(",", kv.Key.Select(h => h.value.ToString()).ToArray()) + "," + kv.Value;
                    }).ToArray()
                );
        }

        private List<List<Hyperparam>> composeHyperParams(Dictionary<Type, Tuple<Hyperparam, Hyperparam>> dicts)
        {
            if (dicts.Count == 0)
                return null;

            //Take out the first
            Type t = dicts.Keys.First();
            Hyperparam first = dicts[t].Item1;
            Hyperparam second = dicts[t].Item2;
            List<Hyperparam> contribution = new List<Hyperparam>();
            for(double i = 0; i <= 1; i+=0.05)
            {
                contribution.Add(Hyperparam.createInstance(t, first.value + (second.value - first.value) * i));
            }

            dicts.Remove(dicts.Keys.First());

            //Get the compositions of the remaining ones
            var res = composeHyperParams(dicts);
            var toReturn = new List<List<Hyperparam>>();

            if (res == null)
            {
                foreach(var c in contribution)
                {
                    var toAdd = new List<Hyperparam>();
                    toAdd.Add(c);
                    toReturn.Add(toAdd);
                }
            }
            else foreach(var vectorToTest in res)
            {
                //vectorToTest is a list of hyperparameter
                List<List<Hyperparam>> newPara = new List<List<Hyperparam>>();
                foreach(var extra in contribution)
                {
                    var toAdd = new List<Hyperparam>(vectorToTest);
                    toAdd.Add(extra);
                    newPara.Add(toAdd);
                }
                toReturn.AddRange(newPara);
            }
            GPUtility.Log(" Composing: Hyperparameters Remaining " + dicts.Count + " Current Number Of Parameters: " + toReturn.Count);
            return toReturn;
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
            //Remove the Jitter term from consideration
            //if (withRespectTo.Equals(typeof(SigmaJ)))
            //    return 0;

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

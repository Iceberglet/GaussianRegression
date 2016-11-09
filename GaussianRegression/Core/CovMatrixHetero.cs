using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace GaussianRegression.Core
{
    //CovMatrix which has input-dependent noises
    //that is obtained via another GP process for its noise
    class CovMatrixHetero : CovMatrix
    {
        CovMatrix matrixForNoise;

        public CovMatrixHetero(CovFunction cf, List<XYPair> list_xy = null, double sigma_f = 1, double delta = 0.0005) : base(cf, list_xy, delta)
        {
            var initialZeroNoise = list_xy.Select(xy => new XYPair(xy.x, sigma_f)).ToList();
            var diag = Enumerable.Repeat(0d, list_xy.Count).ToArray();
            matrixForNoise = new CovMatrix(cf, initialZeroNoise, delta);
            K_diag = Matrix<double>.Build.Diagonal(diag);
            K_base = K_base;
            this.performNoiseAnalysis();
        }

        //Add noise evaluation process
        private new void addX(List<XYPair> pairs)
        {
            base.addX(pairs);
            this.performNoiseAnalysis();
        }

        private static readonly int MAX_HETEROSCEDASTIC_ITERATION = 100;
        private static readonly int HETEROSCEDASTIC_POINT_SAMPLE_SIZE = 20;
        private static readonly double HETEROSCEDASTIC_CONVERGENCE_PERCENTAGE = 0.03;

        //Using Most Likely Heteroscedastic Approach
        //http://www.machinelearning.org/proceedings/icml2007/papers/326.pdf
        public void performNoiseAnalysis()
        {
            int counter = 0;
            double previousNoiseSum = 0;
            bool converged = false;
            Dictionary<Vector<double>, NormalDistribution> resulting_z = new Dictionary<Vector<double>, NormalDistribution>();

            while (counter < MAX_HETEROSCEDASTIC_ITERATION && !converged)
            {
                GPUtility.Log("Heteroscedastic Iter: " + counter);

                //1. Get Empirical Noise at all sampled points on GP_0
                List<XYPair> noise_z = new List<XYPair>();  //Note: the y here refers to the noise term
                List<XYPair> knownPoints = this.xyPairs.ToList();

                Dictionary<Vector<double>, NormalDistribution> dictForSampled = new Dictionary<Vector<double>, NormalDistribution>();
                knownPoints.ForEach(x => {
                    dictForSampled.Add(x.x, this.getPosterior(x.x));
                });

                foreach (XYPair xyPair in knownPoints)
                {
                    NormalDistribution nd = dictForSampled[xyPair.x];  // current estimate
                    double varEstimate = 0;

                    for (int i = 0; i < HETEROSCEDASTIC_POINT_SAMPLE_SIZE; i++)
                    {
                        double sample = Normal.InvCDF(nd.mu, nd.sd, GPUtility.NextProba());
                        varEstimate += Math.Pow((xyPair.y - sample), 2);
                    }
                    varEstimate *= 0.5 / HETEROSCEDASTIC_POINT_SAMPLE_SIZE;
                    varEstimate = Math.Sqrt(varEstimate);   //Back to SD

                    //the new GP is performed on the logarithm of SD - so the SD is always positive
                    varEstimate = Math.Log(varEstimate);
                    noise_z.Add(new XYPair(xyPair.x, varEstimate));
                }

                var nextNoiseSum = noise_z.Sum(n => n.y * n.y);
                var currentError = Math.Abs(previousNoiseSum - nextNoiseSum) / nextNoiseSum;
                if (currentError < HETEROSCEDASTIC_CONVERGENCE_PERCENTAGE)
                {
                    converged = true;
                }
                else
                {
                    previousNoiseSum = nextNoiseSum;
                }
                GPUtility.Log("Current Error " + currentError);

                //2. Construct another Gaussian CovMatrix to evaluate noise
                matrixForNoise = new CovMatrix(cf, noise_z, delta);
                //3. Update the diagonal matrices for this


                this.updateNoise(noise_z);
                
                counter++;
            }
        }

        //Updates the K_diag and predicted noise term for each point
        private void updateNoise(List<XYPair> noise_z)
        {
            Vector<double>[] xInSample = xyPairs.Select(pair => pair.x).ToArray();

            double[,] k_diag = K_diag == null ? new double[K_base.RowCount, K_base.ColumnCount] : K_diag.ToArray();

            foreach (XYPair noise in noise_z)
            {
                int idx = Array.IndexOf(xInSample, noise.x);
                if (idx > -1)
                    k_diag[idx, idx] = Math.Exp(noise.y);
            }
            K_diag = Matrix<double>.Build.DenseOfArray(k_diag);
            K_base = K_base;
            return;
        }

        public override NormalDistribution getPosterior(Vector<double> x_0)
        {
            var res = base.getPosterior(x_0);
            var moreSD = Math.Exp(matrixForNoise.getPosterior(x_0).mu);     //Notice the Exp
            return new NormalDistribution(res.mu, res.sd + moreSD);
        }


    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    class CovMatrix
    {
        public readonly CovFunction cf;
        public XYPair[] xyPairs
        {
            get; private set;
        }
        private Dictionary<Vector<double>, double> noises;

        private Matrix<double> K;
        private Matrix<double> K_base       //A Square Matrix (without input dependent diagonal noise term)
        {
            get { return K_B; }
            set {
                K_B = value;
                if(K_diag == null || K_B.ColumnCount != K_diag.ColumnCount)
                {
                    K_diag = Matrix<double>.Build.Dense(K_B.RowCount, K_B.ColumnCount);
                }
                K = K_B.Add(K_diag);        //Sets K
                K_inverse = K.Inverse();    //Sets K_inverse
            }   //also sets the inverse
        }
        private Matrix<double> K_B;

        //ComputationalHelpers
        private Matrix<double> K_diag;      //Input dependent variance!
        private Matrix<double> K_inverse;

        private double delta;   //For perturbation on sampled points
        public CovMatrix(CovFunction cf, List<XYPair> list_xy = null, double delta = 0.0005)
        {
            this.cf = cf;
            this.delta = delta;
            this.xyPairs = new XYPair[0];
            if (list_xy != null && list_xy.Count > 0)
            {
                addX(list_xy);
            }
        }

        //Expand the matrix
        private void addX(List<XYPair> pairs)
        {
            if (pairs == null || pairs.Count == 0)
                throw new Exception("You are adding 0 new elements to the matrix");

            int originalSize = 0;
            double[,] currentMatrix = null;
            if (K_base != null)
            {
                currentMatrix = K_base.ToArray();
                originalSize = currentMatrix.GetLength(0);
            }

            this.xyPairs = xyPairs.Concat(pairs.ToArray());

            double[,] covValues = new double[originalSize + pairs.Count, originalSize + pairs.Count];

            for (int i = 0; i < originalSize + pairs.Count; ++i)
            {
                for (int j = 0; j < originalSize + pairs.Count; ++j)
                {
                    if (i < originalSize && j < originalSize)
                        covValues[i, j] = currentMatrix[i, j];
                    else
                        covValues[i, j] = cf.eval(xyPairs[i].x, xyPairs[j].x);
                }
            }
            K_base = Matrix<double>.Build.DenseOfArray(covValues);
        }

        public void addX(XYPair xy)
        {
            addX(new XYPair[] { xy }.ToList());
        }

        public NormalDistribution getPosterior(Vector<double> x_0)
        {
            if (xyPairs == null || xyPairs.Length == 0)
                throw new Exception("Cov Matrix is Empty!");

            List<Vector<double>> sampled = xyPairs.Select(xy => xy.x).ToList();
            double mu, sd;
            Vector<double> usable_x_0 = x_0;

            if (sampled.Contains(x_0))
            {
                //TRICK: Perturb it a tiny little bit so that we don't get NAN...
                usable_x_0 = Utility.Perturb(x_0, delta);
            }

            double[,] k_1 = new double[1, xyPairs.Length];      //The CovMatrix between this point and known points
            double[,] k_0 = new double[1, 1];                   //The singleton matrix for this point
            double[,] y = new double[xyPairs.Length, 1];

            for (int i = 0; i < xyPairs.Length; ++i)
            {
                k_1[0, i] = cf.eval(usable_x_0, xyPairs[i].x);
                y[i, 0] = xyPairs[i].y;
            }
            k_0[0, 0] = cf.eval(usable_x_0, usable_x_0);

            Matrix<double> K_1 = Matrix<double>.Build.DenseOfArray(k_1);    //horizontal matrix
            Matrix<double> K_0 = Matrix<double>.Build.DenseOfArray(k_0);    //singleton
            Matrix<double> Y = Matrix<double>.Build.DenseOfArray(y);    //a vertical 1-n matrix

            //intermediate result
            Matrix<double> K_1_multiply_K_inverse = K_1.Multiply(K_inverse);

            mu = K_1_multiply_K_inverse.Multiply(Y).ToArray()[0, 0];

            double k0 = K_0.ToArray()[0, 0];
            double k0_right = K_1_multiply_K_inverse.Multiply(K_1.Transpose()).ToArray()[0, 0];

            sd = K_0.Subtract(K_1_multiply_K_inverse.Multiply(K_1.Transpose())).ToArray()[0, 0];

            if (double.IsNaN(mu) || double.IsInfinity(mu))
            {
                double det_base = K_base.Determinant();
                double det = K.Determinant();
                Matrix<double> k1 = K_1_multiply_K_inverse.Multiply(Y);
                throw new Exception("Unlikely Results!");
            }

            if (double.IsNaN(sd) || double.IsInfinity(sd))
                throw new Exception("Unlikely Results!");

            sd = Math.Sqrt(sd);


            if (noises != null)
                sd += noises[x_0];

            return new NormalDistribution(mu, sd);
        }
        
        //Updates the K_diag and predicted noise term for each point
        public void updateNoise(List<XYPair> noise_z)
        {
            if (noises == null)
                noises = new Dictionary<Vector<double>, double>();

            Vector<double>[] xInSample = xyPairs.Select(pair => pair.x).ToArray();

            double[,] k_diag = K_diag == null ? new double[K_base.RowCount, K_base.ColumnCount] : K_diag.ToArray();

            foreach(XYPair noise in noise_z)
            {
                if (!noises.ContainsKey(noise.x))
                    noises.Add(noise.x, noise.y);
                else noises[noise.x] += noise.y;
                //Update the diagonal matrix
                int idx = Array.IndexOf(xInSample, noise.x);
                if (idx > -1)
                    k_diag[idx, idx] += noise.y;
            }
            K_diag = Matrix<double>.Build.DenseOfArray(k_diag);
            K = K_base.Add(K_diag);
            //double[] sum = K.Diagonal().ToArray();
            return;
        }
    }
}

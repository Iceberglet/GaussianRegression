using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    internal class CovMatrix
    {
        public readonly CovFunction cf;
        public XYPair[] xyPairs
        {
            get; private set;
        }

        protected Matrix<double> K;           //K = K_Base + K_diag
        protected Matrix<double> K_base       //A Square Matrix (without input dependent diagonal noise term)
        {
            get { return K_B; }
            set {
                K_B = value;
                if (value != null)
                {
                    if (K_diag == null || K_B.ColumnCount != K_diag.ColumnCount)
                    {
                        K_diag = Matrix<double>.Build.Dense(K_B.RowCount, K_B.ColumnCount);
                    }
                    K = K_B.Add(K_diag);        //Sets K
                    if (K.Determinant() == 0)
                        throw new Exception("Invalid K: Singular");
                    K_inverse = K.Inverse();    //Sets K_inverse
                }
            }
        }
        protected Matrix<double> K_B;

        //ComputationalHelpers
        protected Matrix<double> K_diag;      //Input dependent variance!
        protected Matrix<double> K_inverse;

        protected double delta;   //For perturbation on sampled points

        //****** Getters ******
        internal Matrix<double> getK()
        {
            return Matrix<double>.Build.DenseOfArray(K.ToArray());
        }
        internal Matrix<double> getY()
        {
            double[,] y = new double[xyPairs.Length, 1];

            for (int i = 0; i < xyPairs.Length; ++i)
            {
                y[i, 0] = xyPairs[i].y;
            }

            return Matrix<double>.Build.DenseOfArray(y);
        }
        internal List<Vector<double>> getX()
        {
            return xyPairs.Select(xy => xy.x).ToList();
        }
        
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
        protected virtual void addX(List<XYPair> pairs)
        {
            //var DebugSampled = xyPairs.Select(xy => xy.x.At(0)).OrderBy(x => x).ToList();
            //var DebugNewX = pairs.Select(xy => xy.x.At(0)).OrderBy(x => x).ToList();

            if (pairs == null || pairs.Count == 0)
                throw new Exception("You are adding 0 new elements to the matrix");

            if(pairs.Any(p => xyPairs.Contains(p)))
                throw new Exception("You are adding a known point! ");

            if (pairs.Distinct().Count() != pairs.Count)
                throw new Exception("You have multiple points with the same X value! ");

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
                        covValues[i, j] = cf.f(xyPairs[i].x, xyPairs[j].x);
                }
            }
            K_base = Matrix<double>.Build.DenseOfArray(covValues);
        }

        public virtual void addX(XYPair xy)
        {
            addX(new XYPair[] { xy }.ToList());
        }

        public void recalculate()
        {
            K_base = null;
            var temp = this.xyPairs;
            this.xyPairs = new XYPair[0];
            addX(temp.ToList());
        }

        public virtual NormalDistribution getPosterior(Vector<double> x_0)
        {
            if (xyPairs == null || xyPairs.Length == 0)
                throw new Exception("Cov Matrix is Empty!");

            List<Vector<double>> sampled = xyPairs.Select(xy => xy.x).ToList();
            double mu, sd;
            Vector<double> usable_x_0 = x_0;
            
            if (sampled.Contains(x_0))
            {
                //TRICK: Perturb it a tiny little bit so that we don't get NAN...
                usable_x_0 = GPUtility.Perturb(x_0, delta);
            }

            double[,] k_1 = new double[1, xyPairs.Length];      //The CovMatrix between this point and known points
            double[,] k_0 = new double[1, 1];                   //The singleton matrix for this point

            for (int i = 0; i < xyPairs.Length; ++i)
            {
                k_1[0, i] = cf.f(usable_x_0, xyPairs[i].x);
            }
            k_0[0, 0] = cf.f(usable_x_0, usable_x_0);

            Matrix<double> K_1 = Matrix<double>.Build.DenseOfArray(k_1);    //horizontal matrix
            Matrix<double> K_0 = Matrix<double>.Build.DenseOfArray(k_0);    //singleton
            Matrix<double> Y = getY();    //a vertical 1-n matrix

            //intermediate result
            Matrix<double> K_1_multiply_K_inverse = K_1.Multiply(K_inverse);

            mu = K_1_multiply_K_inverse.Multiply(Y).ToArray()[0, 0];

            var K_01 = K_1_multiply_K_inverse.Multiply(K_1.Transpose());
            sd = K_0.Subtract(K_01).ToArray()[0, 0];


            //****** Debugging: For Unlikely Results *****
            if (double.IsNaN(mu) || double.IsInfinity(mu))
            {
                double det_base = K_base.Determinant();
                double det = K.Determinant();
                Matrix<double> k1 = K_1_multiply_K_inverse.Multiply(Y);
                throw new Exception("Unlikely Mu Results!");
            }

            if (double.IsNaN(sd) || double.IsInfinity(sd) || sd < 0)
                throw new Exception("Unlikely Sd Results!");

            sd = Math.Sqrt(sd);

            return new NormalDistribution(mu, sd);
        }
        
    }
}

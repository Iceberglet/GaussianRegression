﻿using System;
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
        private XYPair[] xyPairs = new XYPair[0];

        private Matrix<double> K_base   //A Square Matrix
        {
            get { return K_base; }
            set { K_base = value; K_base_inverse = value.Inverse(); }   //also sets the inverse
        }
        private Matrix<double> K_var;   //Input dependent variance!

        //ComputationalHelpers
        private Matrix<double> K_base_inverse;


        public CovMatrix(CovFunction cf, List<XYPair> list_xy = null)
        {
            this.cf = cf;
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

        public XYEstimate getPosterior(Vector<double> x_0)
        {
            if (xyPairs.Length == 0)
                throw new Exception("Cov Matrix is Empty!");

            double[,] k_1 = new double[1, xyPairs.Length];      //The CovMatrix between this point and known points
            double[,] k_0 = new double[1, 1];                   //The singleton matrix for this point
            double[,] y = new double[xyPairs.Length, 1];

            for (int i = 0; i < xyPairs.Length; ++i)
            {
                k_1[0, i] = cf.eval(x_0, xyPairs[i].x);
                y[i, 0] = xyPairs[i].y;
            }
            k_0[0, 0] = cf.eval(x_0, x_0);

            Matrix<double> K_1 = Matrix<double>.Build.DenseOfArray(k_1);
            Matrix<double> K_0 = Matrix<double>.Build.DenseOfArray(k_0);
            Matrix<double> Y = Matrix<double>.Build.DenseOfArray(y);

            Matrix<double> K_1_multiply_K_base_inverse = K_1.Multiply(K_base.Inverse());

            double mu = K_1_multiply_K_base_inverse.Multiply(Y).ToArray()[0, 0];
            double sd = K_0.Subtract(K_1_multiply_K_base_inverse.Multiply(K_1.Transpose())).ToArray()[0, 0];
            return new XYEstimate(x_0, mu, sd);
        }
    }
}

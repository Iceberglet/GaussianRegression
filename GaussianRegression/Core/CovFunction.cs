using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace GaussianRegression.Core
{
    class CovFunction
    {


        public static CovFunction GaussianNoise(double sigma)
        {
            Random rand = new Random();

            Func<Vector<double>, Vector<double>, double> f = (a, b) =>
            {
                if (a.SequenceEqual(b))
                    return sigma * sigma; // * Normal.InvCDF(0, 1, rand.NextDouble());
                else return 0;
            };

            return new CovFunction(f);
        }
        
        public static CovFunction SquaredExponential(double l, double sigmaF)
        {
            double l2 = 2 * l * l;
            double sigma2 = sigmaF * sigmaF;
            Func<Vector<double>, Vector<double>, double> f = (a, b) =>
            {
                double d = (a - b).L2Norm();
                return sigma2 * Math.Exp( d * d / l2);
            };

            return new CovFunction(f);
        }
        
        public static CovFunction operator +(CovFunction f1, CovFunction f2)
        {
            Func<Vector<double>, Vector<double>, double> f = (a, b) =>
            {
                return f1.eval(a, b) + f2.eval(a, b);
            };
            return new CovFunction(f);
        }

        public static CovFunction Matern(double l)
        {
            throw new Exception("Not Implemented");
        }


        // *********** Actual Implementation *************

        //public readonly List<Func<Vector<double>, Vector<double>, double>> f_derivatives;

        public readonly Func<Vector<double>, Vector<double>, double> covarFunction;
        //private int dimension;

        //The burden is on user to ensure f does take two vector of N length
        public CovFunction(Func<Vector<double>, Vector<double>, double> f)//, int dimension)
        {
            this.covarFunction = f;
            //this.dimension = dimension;
        }

        public double eval(Vector<double> a, Vector<double> b)
        {
            //if (a.Count != dimension || b.Count != dimension)
            //    throw new Exception("Invalid input for covariance matrix, dimension mismatch");
            return covarFunction(a, b);
        }
    }
}

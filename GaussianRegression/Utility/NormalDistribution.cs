using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Distributions;

namespace GaussianRegression.Core
{
    //Wrapper Class
    public class NormalDistribution
    {
        public readonly double mu;
        public readonly double sd;

        public NormalDistribution(double m, double s)
        {
            mu = m;
            sd = s;
        }

        public double pdf(double x)
        {
            return Normal.PDF(mu, sd, x);
        }

        public double cdf(double x, bool moreThan = false)
        {
            if (moreThan)
                return 1 - Normal.CDF(mu, sd, x);
            return Normal.CDF(mu, sd, x);
        }

        public override string ToString()
        {
            return "Normal(" + mu + ", " + sd + ")";
        }

        public double getExpectedImprovement(double y_b, bool lessIsBetter = true)
        {
            return GetExpectedImprovement(y_b, this, lessIsBetter);
        }

        private static double fac = 1 / Math.Sqrt(2 * Math.PI);

        public static double GetExpectedImprovement(double y_b, NormalDistribution norm, bool lessIsBetter = true)
        {
            double mu = norm.mu;
            double sd = norm.sd;
            double firstTerm = (y_b - mu) * (lessIsBetter? Normal.CDF(mu, sd, y_b) : Normal.CDF(mu, sd, y_b) - 1);
            double expTerm = (y_b - mu) / sd;
            double secondTerm = sd * fac * Math.Exp(-0.5 * expTerm * expTerm);
            return firstTerm + secondTerm;
        }

        public static NormalDistribution operator +(NormalDistribution n1, NormalDistribution n2)
        {
            return new NormalDistribution(n1.mu + n2.mu, Math.Sqrt(n1.sd * n1.sd + n2.sd * n2.sd));
        }
        
        public static double GetExpectedImprovement(double y_b, NormalDistribution n1, NormalDistribution n2, bool lessIsBetter = true)
        {
            return (n1 + n2).getExpectedImprovement(y_b * 2, lessIsBetter);
        }

        public static NormalDistribution operator *(NormalDistribution n1, NormalDistribution n2)
        {
            double s1 = n1.sd * n1.sd;
            double s2 = n2.sd * n2.sd;
            double sd = Math.Sqrt(s1 * s2 / (s1 + s2));
            double mu = (n1.mu * s2 + n2.sd * s1) / (s1 + s2);
            return new NormalDistribution(mu, sd);
        }
    }
}

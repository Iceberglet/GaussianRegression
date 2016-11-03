using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.Distributions;

namespace GaussianRegression
{
    //Wrapper Class
    class NormalDistribution
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
    }
}

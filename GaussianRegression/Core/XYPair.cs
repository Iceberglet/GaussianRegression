using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    class XYPair
    {
        public readonly Vector<double> x;
        public readonly double y;

        public XYPair(Vector<double> x, double y)
        {
            this.x = x;
            this.y = y;
        }
    }

    class XYEstimate : XYPair
    {
        public readonly double mean;
        public readonly double sd;

        public XYEstimate(Vector<double> x, double mean, double sd) : base(x, 0)
        {
            this.mean = mean;
            this.sd = sd;
        }
    }
}

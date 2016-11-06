using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    public class XYPair
    {
        public readonly Vector<double> x;
        public readonly double y;

        public XYPair(Vector<double> x, double y)
        {
            this.x = x;
            this.y = y;
        }
    }

    public class LabeledVector
    {
        public readonly int idx;
        public Vector<double> x;

        public LabeledVector(int i, Vector<double> x)
        {
            this.idx = i;
            this.x = x;
        }
    }
}

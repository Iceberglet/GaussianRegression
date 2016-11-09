using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace GaussianRegression.Core
{
    public class CovFunction
    {
        public static CovFunction SquaredExponential(LengthScale L, SigmaF SF)
        {
            CovFunction res = null;
            Func<Vector<double>, Vector<double>, double> newF = (a, b) =>
            {
                var l = res == null ? L.value : res.param[typeof(LengthScale)].value;
                var sigma = res == null ? SF.value : res.param[typeof(SigmaF)].value;
                var l2 = 2 * l * l;
                var sigma2 = sigma * sigma;
                double d = (a - b).L2Norm();
                return sigma2 * Math.Exp(-d * d / l2);
            };
            
            Func<Type, Func<Vector<double>, Vector<double>, double>> newDiff = (t) =>
            {
                var l = res == null ? L.value : res.param[typeof(LengthScale)].value;
                var sf = res == null ? SF.value : res.param[typeof(SigmaF)].value;
                var l2 = 2 * l * l;
                if (t == typeof(LengthScale))
                {
                    return (a, b) =>
                    {
                        double d2 = Math.Pow((a - b).L2Norm(), 2);
                        return d2 / Math.Pow(l, 3) * sf * sf * Math.Exp(-d2 / l2);
                    };
                }
                if (t == typeof(SigmaF))
                {
                    return (a, b) =>
                    {
                        double d = (a - b).L2Norm();
                        return 2 * sf * Math.Exp(-d * d / l2);
                    };
                }
                else return (a, b) => 0;
            };

            res = new CovFunction(newF, newDiff);
            res.addParams(L, SF);
            return res;
        }

        public static CovFunction GaussianNoise(SigmaJ SJ)
        {
            CovFunction res = null;
            Func<Vector<double>, Vector<double>, double> newF = (a, b) =>
            {
                var sj = res == null ? SJ.value : res.param[typeof(SigmaJ)].value;
                if (a.SequenceEqual(b))
                {
                    return sj * sj;
                }
                else return 0;
            };

            Func<Type, Func<Vector<double>, Vector<double>, double>> newDiff = (t) =>
            {
                var sj = res == null ? SJ.value : res.param[typeof(SigmaJ)].value;


                if (t == typeof(SigmaJ))
                {
                    return (a, b) =>
                    {
                        if (a.SequenceEqual(b))
                        {
                            return 2 * sj;
                        }
                        else return 0;
                    };
                }
                else return (a, b) => 0;
            };

            res = new CovFunction(newF, newDiff);
            res.addParams(SJ);
            return res;
        }

        public static CovFunction Matern(LengthScale L, Dof D)
        {
            throw new NotImplementedException();
        }

        // *********** Actual Implementation *************

        //public readonly List<Func<Vector<double>, Vector<double>, double>> f_derivatives;

        internal Dictionary<Type, Hyperparam> param;
        internal readonly Func<Vector<double>, Vector<double>, double> f;
        internal readonly Func<Type, Func<Vector<double>, Vector<double>, double>> differential;

        internal void addParams(params Hyperparam[] param)
        {
            foreach(var i in param)
            {
                this.param[i.type] =  i;
            }
        }
        
        private CovFunction(Func<Vector<double>, Vector<double>, double> f,
            Func<Type, Func<Vector<double>, Vector<double>, double>> diff
            )
        {
            this.param = new Dictionary<Type, Hyperparam>();
            this.f = f;
            this.differential = diff;
        }

        //Combine two CovFunctions
        public static CovFunction operator +(CovFunction f1, CovFunction f2)
        {
            Func<Vector<double>, Vector<double>, double> newF = (a, b) =>
            {
                return f1.f(a, b) + f2.f(a, b);
            };
            Func<Type, Func<Vector<double>, Vector<double>, double>> newDiff = (t) => (a, b) =>
            {
                /*
                var left = f1.differential(t)(a, b);
                var right = f2.differential(t)(a, b);
                if (left * right != 0)
                    throw new Exception("Gotcha!");*/
                return f1.differential(t)(a, b) + f2.differential(t)(a, b);
            };
            var res = new CovFunction(newF, newDiff);
            foreach (var hyper in f1.param)
                res.addParams(hyper.Value);
            foreach (var hyper in f2.param)
                res.addParams(hyper.Value);
            //Repointing the subfunction param pointer
            f1.param = res.param;
            f2.param = res.param;
            return res;
        }
    }
}

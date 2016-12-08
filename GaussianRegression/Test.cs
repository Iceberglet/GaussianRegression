using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Statistics;
using GaussianRegression.Core;

namespace GaussianRegression
{
    public class Solution
    {
        public double LFValue { get; set; }
        public double HFValue { get; set; }
        public int LFRank { get; set; }
        public int HFRank { get; set; }
        public double proba { get; set; }
        public double a { get; set; }
        public double b { get; set; }
        public double c { get; set; }
    }

    static class Test
    {
        static Random rand = new Random();

        public static void testXu2014(int gg)
        {
            Func<List<Solution>, List<Solution>> RankAndSort = (solutions) =>
            {
                int rank;
                rank = 0; foreach (var s in solutions.OrderBy(s => s.HFValue)) s.HFRank = rank++;
                rank = 0; foreach (var s in solutions.OrderBy(s => s.LFValue)) s.LFRank = rank++;
                return solutions.OrderBy(s => s.LFRank).ToList();
            };

            Func<int, List<Solution>> Xu2014 = (g) =>
            {
                var solutions = new List<Solution>();
                for (var x = 0.0; x <= 100; x += 0.1)
                {
                    double lfValue = 0;
                    switch (g)
                    {
                        case 1: lfValue = -(Math.Pow(Math.Sin(0.09 * Math.PI * x), 6) / Math.Pow(2, 2 * Math.Pow((x - 10) / 80, 2))); break; // g1
                        case 2: lfValue = -(Math.Pow(Math.Sin(0.09 * Math.PI * (x - 1.2)), 6) / Math.Pow(2, 2 * Math.Pow((x - 10) / 80, 2))); break; // g2
                        case 3: lfValue = -(Math.Pow(Math.Sin(0.09 * Math.PI * (x - 5)), 6) / Math.Pow(2, 2 * Math.Pow((x - 10) / 80, 2))); break; // g3
                    }
                    solutions.Add(new Solution
                    {
                        HFValue = -(Math.Pow(Math.Sin(0.09 * Math.PI * x), 6) / Math.Pow(2, 2 * Math.Pow((x - 10) / 80, 2)) +
                                0.1 * Math.Cos(0.5 * Math.PI * x) + 0.5 * Math.Pow((x - 40) / 60, 2) + 0.4 * Math.Sin((x + 10) / 100 * Math.PI)),
                        LFValue = lfValue,
                    });
                }
                return RankAndSort(solutions);
            };


            var num = 50;
            var sols = Xu2014(gg);
            var sampled = new List<Solution>();
            
            for (int i = 0; i < num; i++)
            {
                var idx = sols.Count / num * i + rand.Next(1, sols.Count / num);
                sampled.Add(sols.ElementAt(idx));
            }
            
            var initial = sampled.Select(s => new XYPair(GPUtility.V(s.LFRank), s.HFValue)).ToList();
            var list_x = sols.Select(s => new LabeledVector(s.LFRank, GPUtility.V(s.LFRank))).ToList();
            var myGP = new GP(initial, list_x, 
                    CovFunction.SquaredExponential(new LengthScale(100), new SigmaF(0.3)) + CovFunction.GaussianNoise(new SigmaJ(0.05)),
                    heteroscedastic: true, estimateHyperPara: true
                    );
            var res = myGP.predict();

            /*
            for (int i = 0; i < num; i++)
            {
                var idx = sols.Count / num * i;
                var sol = sols.ElementAt(idx);
                if (sampled.Contains(sol))
                {
                    GPUtility.Log("Failed to add a sample", GPUtility.LogLevel.DEBUG);
                    continue;
                }
                sampled.Add(sol);
                myGP.addPoint(new XYPair(GPUtility.V(sol.LFRank), sol.HFValue));
                res = myGP.predict();
            }*/


            FileService fs = new FileService("TestXu2014.csv");

            fs.writeToFile(FileService.convertGPResult(res, initial));
        }
        public static void testSimple()
        {
            List<XYPair> values = new List<XYPair>();
            List<LabeledVector> list_x = new List<LabeledVector>();
            for(int i = 1; i < 500; i++)
            {
                double xx = i / 10.0;
                Vector<double> x = GPUtility.V(xx);
                list_x.Add(new LabeledVector(0, x));
                if (i % 20 == 0)
                    values.Add(new XYPair(x, xx * xx + (rand.NextDouble() - 1) * 500));
            }

            CovFunction cf = CovFunction.SquaredExponential(new LengthScale(8), new SigmaF(50)) + CovFunction.GaussianNoise(new SigmaJ(50));
            
            GP myGP = new GP(sampledValues: values, list_x: list_x.ToList(), cov_f: cf,
                heteroscedastic: false, estimateHyperPara: true
                );
            var res = myGP.predict();

            FileService fs = new FileService("Test.csv");

            fs.writeToFile(FileService.convertGPResult(res, values));
        }
        public static void testHyperparameterEstimation()
        {
            List<XYPair> values = FileService.readFromFile("Motor.txt", separator: '\t');
            CovFunction cf = CovFunction.SquaredExponential(new LengthScale(8), new SigmaF(20)) + CovFunction.GaussianNoise(new SigmaJ(1));
            CovMatrix covMatrix = new CovMatrix(cf, values);
            //TODO: Replace the nulls
            ModelOptimizer mo = new ModelOptimizer(covMatrix, cf, null, null);

            //Add testing limits
            Dictionary<Type, Tuple<Hyperparam, Hyperparam>> dict = new Dictionary<Type, Tuple<Hyperparam, Hyperparam>>();
            dict.Add(typeof(LengthScale), new Tuple<Hyperparam, Hyperparam>(new LengthScale(1), new LengthScale(20)));
            dict.Add(typeof(SigmaF), new Tuple<Hyperparam, Hyperparam>(new SigmaF(5), new SigmaF(35)));
            dict.Add(typeof(SigmaJ), new Tuple<Hyperparam, Hyperparam>(new LengthScale(0.1), new LengthScale(10)));

            mo.evaluateLog(dict);
            
        }
        public static void testMotor()
        {
            List<XYPair> values = FileService.readFromFile("Motor.txt", separator : '\t');
            List<Vector<double>> list_x = new List<Vector<double>>();
            //x value from 2 to 60
            for (int i = 20; i < 600; i++)
            {
                double x = i / 10.0 + 0.05;
                list_x.Add(GPUtility.V(x));
            }
            
            var SF = Math.Sqrt(Statistics.Variance(values.Select(xy => xy.y))) / 2;
            //Utility.Log("Variance determined as: " + SF);

            CovFunction cf = CovFunction.SquaredExponential(new LengthScale(5), new SigmaF(10)) + CovFunction.GaussianNoise(new SigmaJ(0.5));

            GP myGP = new GP(sampledValues: values, list_x: list_x.Select(x => new LabeledVector(0, x)).ToList(), cov_f: cf,
                heteroscedastic: false, estimateHyperPara: true
                );
            var res = myGP.predict();

            FileService fs = new FileService("Test-Motor.csv");

            fs.writeToFile(FileService.convertGPResult(res, values));
        }

        public static void testComplex()
        {
            Func<double, double> f_pure = x => -(x - 134) * (x - 167) / 100.0 - 1000;
            Func<double, double> f_sd = x => 60 + Math.Exp(x / 100);
            Func<double, double> f = x => f_pure(x) + Normal.InvCDF(0, f_sd(x), rand.NextDouble());

            int size = 600;
            List<XYPair> sampled = new List<XYPair>();
            List<XYPair> xyPairs = new List<XYPair>();
            Vector<double>[] list_x = new Vector<double>[size];

            for (int i = 0; i < size; ++i)
            {
                Vector<double> newX = GPUtility.V(i);

                list_x[i] = newX;
                double y = f(i);
                XYPair newPair = new XYPair(newX, y);

                xyPairs.Add(newPair);
                if (rand.NextDouble() < 0.15)
                    sampled.Add(newPair);
            }
            CovFunction cf = CovFunction.SquaredExponential(new LengthScale(20), new SigmaF(1)) + CovFunction.GaussianNoise(new SigmaJ(0.1));

            GP myGP = new GP(sampledValues: sampled, list_x: list_x.Select(x => new LabeledVector(0, x)).ToList(), cov_f: cf,
                heteroscedastic: true, estimateHyperPara: true
                );
            var res = myGP.predict();

            FileService fs = new FileService("Test.csv");
            
            /*
            CovMatrix covMatrix = new CovMatrix(cf, sampled);
            Dictionary<Vector<double>, NormalDistribution> res = new Dictionary<Vector<double>, NormalDistribution>();
            xyPairs.ForEach(xy => {
                res.Add(xy.x, covMatrix.getPosterior(xy.x));
            });*/

            fs.writeToFile(FileService.convertGPResult(res, sampled));
        }


        public static void testCovFunc()
        {
            CovFunction cf2 = CovFunction.SquaredExponential(new LengthScale(8), new SigmaF(50)) + CovFunction.GaussianNoise(new SigmaJ(1));
            //CovFunction cf2 = CovFunction.GaussianNoise(10);

            Vector<double> a = Vector<double>.Build.Dense(new double[] { 1, 3, 5, 7 });
            Vector<double> a_prime = Vector<double>.Build.Dense(new double[] { 1, 3, 5, 7 });
            Vector<double> b = Vector<double>.Build.Dense(new double[] { 2, 4, 6, 8 });
            //Vector<double> c = Vector<double>.Build.Dense(new double[] { 2, 4, 6, 8, 10 });

            Console.WriteLine("a and a_prime gives: " + cf2.f(a, a_prime));
            Console.WriteLine("a and b gives: " + cf2.f(a, b));
            Console.WriteLine("a and b differential w.r.t LengthScale: " + cf2.differential(typeof(LengthScale))(a, b));
            Console.WriteLine("a and b differential w.r.t SigmaF: " + cf2.differential(typeof(SigmaF))(a, b));
            Console.WriteLine("a and b differential w.r.t SigmaJ: " + cf2.differential(typeof(SigmaJ))(a, b));
            Console.WriteLine("a and a_prime differential w.r.t SigmaJ: " + cf2.differential(typeof(SigmaJ))(a, a_prime));
            //Console.WriteLine("a and c gives: " + cf2.f(a, c));
        }
    }
}

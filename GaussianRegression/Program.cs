using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using GaussianRegression.Core;

namespace GaussianRegression
{
    class Program
    {
        static Random rand = new Random();
        static Func<double, double> f_pure = x => -(x - 134) * (x - 167) / 100.0 + 250;
        static Func<double, double> f_sd = x => 50 + Math.Exp(x / 100);
        static Func<double, double> f = x => f_pure(x) + Normal.InvCDF(0, f_sd(x), rand.NextDouble());

        static void Main(string[] args)
        {
            int size = 600;
            List<XYPair> sampled = new List<XYPair>();
            List<XYPair> xyPairs = new List<XYPair>();
            Vector<double>[] list_x = new Vector<double>[size];

            for (int i = 0; i < size; ++i)
            {
                Vector<double> newX = Utility.V(i);

                list_x[i] = newX;
                double y = f(i);
                XYPair newPair = new XYPair(newX, y);

                xyPairs.Add(newPair);
                //if (rand.NextDouble() < 0.15)
                sampled.Add(newPair);
            }
            CovFunction cf = CovFunction.SquaredExponential(20, 60) + CovFunction.GaussianNoise(10);

            GP myGP = new GP(sampledValues: sampled, list_x: list_x.ToList(), cov_f: cf,
                heteroscedastic : true,
                lengthScale : 60, sigma_f : 20);

            //List<int> xs = xyPairs.Select(pair => (int)pair.x.ToArray()[0]).ToList();
            FileService fs = new FileService("Test.csv");
            //FileService fs2 = new FileService("Sample.csv");
            //fs2.writeToFile(xyPairs.Select(xy => xy.x.toString() + "," + xy.y).ToArray());

            
            //var res = myGP.predict();
            CovMatrix covMatrix = new CovMatrix(cf, sampled);
            Dictionary<XYPair, NormalDistribution> res = new Dictionary<XYPair, NormalDistribution>();
            xyPairs.ForEach(xy => {
                res.Add(xy, covMatrix.getPosterior(xy.x));
            });

            fs.writeToFile(FileService.convertGPResult(res));
            



            //Console.ReadLine();

            /*** Tested ***
            CovFunction cf2 = CovFunction.SquaredExponential(10);
            //CovFunction cf2 = CovFunction.GaussianNoise(10);

            Vector<double> a = Vector<double>.Build.Dense(new double[] { 1, 3, 5, 7 });
            Vector<double> a_prime = Vector<double>.Build.Dense(new double[] { 1, 3, 5, 7 });
            Vector<double> b = Vector<double>.Build.Dense(new double[] { 2, 4, 6, 8 });
            Vector<double> c = Vector<double>.Build.Dense(new double[] { 2, 4, 6, 8, 10 });

            Console.WriteLine("a and a_prime gives: " + cf2.eval(a, a_prime));
            Console.WriteLine("a and b gives: " + cf2.eval(a, b));
            Console.WriteLine("a and c gives: " + cf2.eval(a, c));
            
            //*/
        }
    }
}

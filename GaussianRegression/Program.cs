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
        static Func<double, double> f = x => f_pure(x) + Normal.InvCDF(0, 10, rand.NextDouble());

        static void Main(string[] args)
        {
            List<XYPair> xyPairs = new List<XYPair>();
            List<int> usedX = new List<int>();
            for(int i = 0; i < 15; ++i)
            {
                int x = 20 * i + rand.Next(-2, 2);
                if (usedX.Contains(x))
                    continue;
                usedX.Add(x);

                xyPairs.Add(new XYPair(
                    Utility.V(x),
                    f(x)
                    )
                );
            }
            CovMatrix m = new CovMatrix(CovFunction.SquaredExponential(20, 10) + CovFunction.GaussianNoise(2.2), xyPairs);

            //List<int> xs = xyPairs.Select(pair => (int)pair.x.ToArray()[0]).ToList();
            FileService fs = new FileService("Test.csv");

            for(int x = 0; x < 300; ++x)
            {
                if (!usedX.Contains(x))
                {
                    XYEstimate xyEstimate = m.getPosterior(Utility.V(x));
                    NormalDistribution nd = new NormalDistribution(xyEstimate.mean, xyEstimate.sd);
                    string newLine = x + "," + (nd.mu - nd.sd * 2) + "," + (nd.mu + nd.sd * 2) + "," +  f_pure(x);
                    fs.writeToFile(newLine);
                }
            }




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

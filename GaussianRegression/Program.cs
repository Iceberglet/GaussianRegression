using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using GaussianRegression.Core;

namespace GaussianRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            Test.testComplex();

            Console.WriteLine("End of Execution.");
            //Console.ReadLine();
        }
    }
}

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
            //Test.testComplex();
            //Test.testXu2014(2);
            Test.testMotor();
            //Test.testSimple();
            //Test.testHyperparameterEstimation();
            //Test.testAnother();
            /*
            var x1 = GPUtility.V(1);
            var x2 = GPUtility.V(0);

            CovFunction cf = CovFunction.SquaredExponential(new LengthScale(1), new SigmaF(1)) + CovFunction.GaussianNoise(new SigmaJ(1));
            Utility.Log("Before: L: " + cf.param[typeof(LengthScale)].value);
            Utility.Log("Before: F: " + cf.f(x1, x2));
            cf.param[typeof(LengthScale)] = Hyperparam.createInstance(typeof(LengthScale), 20);
            cf.param[typeof(SigmaF)] = Hyperparam.createInstance(typeof(SigmaF), 20);
            cf.param[typeof(SigmaJ)] = Hyperparam.createInstance(typeof(SigmaJ), 20);
            Utility.Log("After: L: " + cf.param[typeof(LengthScale)].value);
            Utility.Log("After: F: " + cf.f(x1, x2));*/
            

            /*
            Console.WriteLine(nd.getExpectedImprovement(-5, false));
            Console.WriteLine(nd.getExpectedImprovement(0, false));
            Console.WriteLine(nd.getExpectedImprovement(5, false));
            */
            Console.WriteLine("End of Execution.");
            Console.ReadLine();
        }
    }
}

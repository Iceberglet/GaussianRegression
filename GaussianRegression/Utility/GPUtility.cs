using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    public static class GPUtility
    {
        private static Random rand = new Random();

        //**************  Random Numbers ********************
        public static double NextProba()
        {
            double res = 0;
            while(res == 0)
            {
                res = rand.NextDouble();
            }
            return res;
        }

        //**************  Extension Methods  ************************
        public static string toString<T>(this IList<T> list, string separator)
        {
            return string.Join(separator, list);
        }

        public static string toString<T>(this Matrix<T> mat) where T : struct, IEquatable<T>, IFormattable
        {
            return "Matrix (" + mat.RowCount + " * " + mat.ColumnCount + ")";
        }

        public static string toString<T>(this Vector<T> v) where T : struct, IEquatable<T>, IFormattable
        {
            if (v.Count == 1)
                return v.ToList().First().ToString();
            else return "[" + string.Join(" ", v.ToList()) + "]";
        }

        public static T[] Concat<T>(this T[] arr, T[] another)
        {
            //T[] res = new T[arr.Length + another.Length];
            List<T> r = arr.ToList();
            r.AddRange(another.ToList());
            return r.ToArray();
        }
        
        //*************** Utility Methods **********************
        //Initializes a V vector
        public static Vector<double> V(params double[] values)
        {
            if (values.Length == 0)
                throw new Exception("You need at least ONE parameter! ");
            return Vector<double>.Build.Dense(values);
        }

        public static Vector<double> Perturb(Vector<double> v, double scale = 0)
        {
            double[] values = v.ToArray();
            for(int i = 0; i < values.Length; i++)
            {
                double d = scale == 0 ? 0.01 * values[i] : scale;
                values[i] += d * Math.Sign(rand.NextDouble() - 0.5);
            }
            return Vector<double>.Build.Dense(values);
        }

        //combines two lists
        public static string[] PairwiseAdd(this string[] a, string[] b, string separator = ",")
        {
            if (a.Length != b.Length)
                throw new Exception("Length Mismatch: " + a.Length + " " + b.Length);
            
            for(int i = 0; i < a.Length; i++)
            {
                a[i] += separator + b[i];
            }
            return a;
        }


        private static readonly bool login = true;
        public static void Log(string s)
        {
            if (login)
                Console.WriteLine(s);
        }


    }
}

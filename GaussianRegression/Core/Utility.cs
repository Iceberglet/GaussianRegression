using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace GaussianRegression.Core
{
    static class Utility
    {
        public static string ToString<T>(this IList<T> list, string separator)
        {
            return string.Join(separator, list);
        }

        public static T[] Concat<T>(this T[] arr, T[] another)
        {
            //T[] res = new T[arr.Length + another.Length];
            List<T> r = arr.ToList();
            r.AddRange(another.ToList());
            return r.ToArray();
        }
        
        public static Vector<double> V(params double[] values)
        {
            return Vector<double>.Build.Dense(values);
        }

        public static string ToString(this Vector<double> v)
        {
            if (v.Count == 1)
                return v.ToList().First().ToString();
            else return "[" + string.Join(" ", v.ToList()) + "]";
        }
    }
}

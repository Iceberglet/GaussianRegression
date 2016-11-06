using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

using GaussianRegression.Core;

namespace GaussianRegression
{
    class FileService
    {
        private readonly string path;

        public FileService(string path)
        {
            this.path = path;
        }

        public void writeToFile(string[] lines, bool append = false)
        {
            if (!File.Exists(path))
            {
                createFile();
            }

            if (!append)
                createFile();

            lines.ToList().ForEach(l => appendLine(l));
        }

        public void writeToFile(string line, bool append = true)
        {
            if (!File.Exists(path))
            {
                createFile();
            }

            if (!append)
                createFile();

            appendLine(line);
        }

        private void appendLine(string s)
        {
            using (StreamWriter sw = File.AppendText(path))
            {
                sw.WriteLine(s);
            }
        }

        private void createFile()
        {
            using (StreamWriter sw = File.CreateText(path)) { sw.Write(""); }
        }

        public static string[] convertGPResult(Dictionary<LabeledVector, NormalDistribution> vars, List<XYPair> sampled)
        {
            var xSampled = new Dictionary<Vector<double>, double>();
            sampled.ForEach(xy => xSampled.Add(xy.x, xy.y));

            var xAndString = new Dictionary<Vector<double>, string>();
            
            foreach(var kv in vars)
            {
                double upper = kv.Value.mu + 1.96 * kv.Value.sd;
                double lower = kv.Value.mu - 1.96 * kv.Value.sd;
                string s = lower + "," + upper; // + "," + kv.Key.y;
                xAndString.Add(kv.Key.x, s);
            }

            foreach(var kv in xSampled)
            {
                if (xAndString.ContainsKey(kv.Key))
                    xAndString[kv.Key] += "," + kv.Value;
                else xAndString.Add(kv.Key, ",," + kv.Value);
            }

            var res = xAndString.Select(kv => kv.Key.toString() + "," + kv.Value).ToList();

            res.Insert(0, ",Lower,Upper,Sampled");

            return res.ToArray();
        }

        public static List<XYPair> readFromFile(string fileName, int xSize = 1, char separator = ' ')
        {
            //Console.WriteLine(Path.GetTempPath());
            //Console.WriteLine(Directory.GetCurrentDirectory());
            string path = Path.Combine(Directory.GetCurrentDirectory(), fileName);
            
            String input_grid = File.ReadAllText(path);
            List<XYPair> xy = new List<XYPair>();
            foreach (var row in input_grid.Split('\n'))
            {
                List<double> x = new List<double>();
                int xIdx = 0;
                foreach (var col in row.Trim().Split(separator))
                {
                    if (xIdx < xSize)
                    {
                        x.Add(double.Parse(col.Trim()));
                        xIdx++;
                        continue;
                    }
                    else
                    {
                        double y = double.Parse(col.Trim());
                        xy.Add(new XYPair(Vector<double>.Build.DenseOfEnumerable(x), y));
                        break;
                    }
                }
            }
            return xy;
        }
    }
}

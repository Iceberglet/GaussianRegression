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

        public static string[] convertGPResult(Dictionary<XYPair, NormalDistribution> vars)
        {
            var res = vars.Select(kv =>
            {
                double upper = kv.Value.mu + 1.96 * kv.Value.sd;
                double lower = kv.Value.mu - 1.96 * kv.Value.sd;
                return kv.Key.x.toString() + "," + kv.Key.y + "," + lower + "," + upper;
            }).ToList();

            res.Insert(0, ",Upper,Lower,Actual");
            return res.ToArray();
        }
    }
}

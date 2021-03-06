﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Statistics;

namespace GaussianRegression.Core
{
    class GP
    {
        private bool heteroscedastic;
        private bool estimateHyperPara;
        private double lengthScale;
        private double sigma_f;
        private double sigma_jitter;
        
        private List<Vector<double>> list_x;

        private readonly CovFunction cov_f;
        private CovMatrix covMatrix;

        public GP(List<XYPair> sampledValues, List<Vector<double>> list_x, CovFunction cov_f,
            bool estimateHyperPara = false, bool heteroscedastic = false,           //configs
            double lengthScale = 1, double sigma_f = 1, double sigma_jitter = 1,       //hyper parameters
            double delta = 0.0005
            )
        {
            this.list_x = list_x;
            this.estimateHyperPara = estimateHyperPara;
            this.heteroscedastic = heteroscedastic;

            this.lengthScale = lengthScale;
            this.sigma_f = sigma_f;
            if (heteroscedastic)
                this.sigma_f = Statistics.StandardDeviation(sampledValues.Select(xy => xy.y)) / 10;
            this.sigma_jitter = sigma_jitter;

            this.cov_f = cov_f;

            if (heteroscedastic)
                this.covMatrix = new CovMatrixHetero(cov_f, sampledValues, sigma_f, delta);
            else this.covMatrix = new CovMatrix(cov_f, sampledValues, delta);

            if (estimateHyperPara)
            {
                ModelOptimizer mo = new ModelOptimizer(covMatrix, cov_f);
                mo.optimize();
            }

            if (heteroscedastic)
                ((CovMatrixHetero)covMatrix).performNoiseAnalysis();

            Utility.Log("Final Hypers: " + string.Join(", ", cov_f.param.Select(kv => kv.Value.value).ToArray()));
        }

        //NOTE must be set null after every time GP is modified
        private Dictionary<Vector<double>, NormalDistribution> lastPredict = null;

        public void addPoint(XYPair newPair)
        {
            covMatrix.addX(newPair);
            lastPredict = null;
        }

        public Dictionary<Vector<double>, NormalDistribution> predict()
        {
            if (lastPredict == null)
            {
                lastPredict = new Dictionary<Vector<double>, NormalDistribution>();
                list_x.ForEach(x => lastPredict.Add(x, covMatrix.getPosterior(x)));
            }
            return lastPredict;
        }
    }
}

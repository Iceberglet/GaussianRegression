using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Statistics;

namespace GaussianRegression.Core
{
    public class GP
    {
        private bool heteroscedastic;
        private bool estimateHyperPara;
        
        private List<LabeledVector> list_x;

        private readonly CovFunction cov_f;
        private CovMatrix covMatrix;
        private ModelOptimizer mo;

        public GP(List<XYPair> sampledValues, List<LabeledVector> list_x, CovFunction cov_f,
            List<Hyperparam> minBounds = null, List<Hyperparam> maxBounds = null,
            bool estimateHyperPara = false, bool heteroscedastic = false, double delta = 0.005           //configs
            )
        {
            this.list_x = list_x;
            this.estimateHyperPara = estimateHyperPara;
            this.heteroscedastic = heteroscedastic;
            
            var sigma_f = Statistics.StandardDeviation(sampledValues.Select(xy => xy.y));

            this.cov_f = cov_f;

            if (heteroscedastic)
            {
                var mean = Statistics.Mean(sampledValues.Select(xy => xy.y));
                var variational_sd = Statistics.StandardDeviation(sampledValues.Select(xy => Math.Abs(xy.y - mean)));
                this.covMatrix = new CovMatrixHetero(cov_f, sampledValues, variational_sd, delta);
            }
            else this.covMatrix = new CovMatrix(cov_f, sampledValues, delta);

            //if (heteroscedastic)
            //    ((CovMatrixHetero)covMatrix).performNoiseAnalysis();
            if (estimateHyperPara)
            {
                if (minBounds == null)
                    minBounds = new List<Hyperparam>();
                if (maxBounds == null)
                    maxBounds = new List<Hyperparam>();
                initializeHyperparameterBounds(minBounds, maxBounds, sampledValues);
                mo = new ModelOptimizer(covMatrix, cov_f, minBounds, maxBounds);
                mo.optimize();
            }

            covMatrix.recalculate();

            GPUtility.Log("Final Hypers: " + string.Join(", ", cov_f.param.Select(kv => kv.Value.value).ToArray()), GPUtility.LogLevel.DEBUG);

            if (heteroscedastic)
            {
                ((CovMatrixHetero)covMatrix).performNoiseAnalysis();

                //For Debug Purpose
                //((CovMatrixHetero)covMatrix).evaluateHeteroResult(list_x);
            }

        }

        private void initializeHyperparameterBounds(List<Hyperparam> minBounds, List<Hyperparam> maxBounds, List<XYPair> sampledValues)
        {

            if (minBounds.Find(b => b.type.Equals(typeof(LengthScale))) == null)
                minBounds.Add(Hyperparam.createInstance(typeof(LengthScale), 0.1));
            if (minBounds.Find(b => b.type.Equals(typeof(SigmaJ))) == null)
                minBounds.Add(Hyperparam.createInstance(typeof(SigmaJ), 0.001));
            if (minBounds.Find(b => b.type.Equals(typeof(SigmaF))) == null)
                minBounds.Add(Hyperparam.createInstance(typeof(SigmaF), 0.001));
            if (maxBounds.Find(b => b.type.Equals(typeof(LengthScale))) == null)
                maxBounds.Add(Hyperparam.createInstance(typeof(LengthScale), 1000));
            if (maxBounds.Find(b => b.type.Equals(typeof(SigmaJ))) == null)
                maxBounds.Add(Hyperparam.createInstance(typeof(SigmaJ), 200));
            if (maxBounds.Find(b => b.type.Equals(typeof(SigmaF))) == null)
                maxBounds.Add(Hyperparam.createInstance(typeof(SigmaF), 1000));
        }

        //NOTE must be set null after every time GP is modified
        private Dictionary<LabeledVector, NormalDistribution> lastPredict = null;

        private static readonly int RE_OPTIMIZE_THRESHOLD = 15;
        int current_optimize_number = RE_OPTIMIZE_THRESHOLD;
        public void addPoint(XYPair newPair)
        {
            covMatrix.addX(newPair);
            lastPredict = null;
            if(current_optimize_number == RE_OPTIMIZE_THRESHOLD)
            {
                current_optimize_number = 0;
            } else
            {
                ((CovMatrixHetero)covMatrix).optimize();
                current_optimize_number++;
            }
        }

        public Dictionary<LabeledVector, NormalDistribution> predict()
        {
            if (lastPredict == null)
            {
                lastPredict = new Dictionary<LabeledVector, NormalDistribution>();
                list_x.ForEach(x => lastPredict.Add(x, covMatrix.getPosterior(x.x)));
            }
            return lastPredict;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

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

        private CovMatrix covMatrix;

        public GP(List<XYPair> sampledValues, List<Vector<double>> list_x, CovFunction f,
            bool estimateHyperPara = false, bool heteroscedastic = false,           //configs
            double lengthScale = 1, float sigma_f = 1, float sigma_jitter = 1       //hyper parameters
            )
        {
            this.list_x = list_x;
            this.estimateHyperPara = estimateHyperPara;
            this.heteroscedastic = heteroscedastic;
            this.lengthScale = lengthScale;
            this.sigma_f = sigma_f;
            this.sigma_jitter = sigma_jitter;

            this.covMatrix = new CovMatrix(f, sampledValues);
        }

        //NOTE must be set null after every time GP is modified
        private List<XYEstimate> lastPredict = null;
        private static readonly int MAX_HETEROSCEDASTIC_ITERATION = 20;
        private static readonly double HETEROSCEDASTIC_CONVERGENCE_PERCENTAGE = 0.1;


        //Using Most Likely Heteroscedastic Approach
        //http://www.machinelearning.org/proceedings/icml2007/papers/326.pdf
        private void updateInputDependentVariance()
        {
            //Each iteration, use prediction to calculate empirical noise at each sampled point
            //These empirical noise are fed to another GP to give a GP for noise
        }

        public List<XYEstimate> predict()
        {
            if (lastPredict == null)
            {
                lastPredict = new List<XYEstimate>();
                list_x.ForEach(x => lastPredict.Add(covMatrix.getPosterior(x)));
                if (heteroscedastic)
                {
                    updateInputDependentVariance();
                }
            }
            return lastPredict;
        }
    }
}

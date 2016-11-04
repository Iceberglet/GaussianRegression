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
    class GP
    {
        private static Random rand = new Random();
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
            this.covMatrix = new CovMatrix(cov_f, sampledValues, delta);
        }

        //NOTE must be set null after every time GP is modified
        private Dictionary<Vector<double>, NormalDistribution> lastPredict = null;
        private static readonly int MAX_HETEROSCEDASTIC_ITERATION = 2;
        private static readonly int HETEROSCEDASTIC_POINT_SAMPLE_SIZE = 100;
        //private static readonly double HETEROSCEDASTIC_CONVERGENCE_PERCENTAGE = 0.1;


        //Using Most Likely Heteroscedastic Approach
        //http://www.machinelearning.org/proceedings/icml2007/papers/326.pdf
        private void updateInputDependentVariance()
        {
            covMatrix.updateNoise(list_x.Select(x => new XYPair(x, sigma_f)).ToList());
            int counter = 0;
            bool converged = false;
            Dictionary<Vector<double>, NormalDistribution> resulting_z = new Dictionary<Vector<double>, NormalDistribution>();
            
            while (counter < MAX_HETEROSCEDASTIC_ITERATION || converged )
            {
                Utility.Log("Heteroscedastic Iter: " + counter);

                //1. Get Empirical Noise at all sampled points on GP_0
                List<XYPair> noise_z = new List<XYPair>();  //Note: the y here refers to the noise term
                List<XYPair> knownPoints = covMatrix.xyPairs.ToList();

                Dictionary<Vector<double>, NormalDistribution> dictForSampled = new Dictionary<Vector<double>, NormalDistribution>();
                knownPoints.ForEach(x => {
                    Vector<double> xx = x.x;
                    NormalDistribution nd = covMatrix.getPosterior(x.x);
                    dictForSampled.Add(x.x, nd);
                    });
                Utility.Log("Done New prediction.");

                foreach (XYPair xyPair in knownPoints)
                {
                    NormalDistribution nd = dictForSampled[xyPair.x];  // current estimate
                    double varEstimate = 0;
                    
                    for(int i = 0; i < HETEROSCEDASTIC_POINT_SAMPLE_SIZE; i++)
                    {
                        double sample = Normal.InvCDF(nd.mu, nd.sd, rand.NextDouble());
                        varEstimate += Math.Pow((xyPair.y - sample), 2);
                    }
                    varEstimate *= 0.5 / HETEROSCEDASTIC_POINT_SAMPLE_SIZE;
                    varEstimate = Math.Sqrt(varEstimate);   //Back to SD
                    
                    //the new GP is performed on the logarithm of SD - so the SD is always positive
                    varEstimate = Math.Log(varEstimate);
                    noise_z.Add(new XYPair(xyPair.x, varEstimate));
                }

                //*******************************************For Debugging
                FileService fs = new FileService("GP_On_Noise_" + counter + ".csv");
                /*
                string[] noises = noise_z.OrderBy(z => z.x.Norm(1)).Select(z => z.x.toString() + "," + z.y).ToArray();
                fs.writeToFile(noises);*/


                //2. Construct another Gaussian Process, GP_1 to evaluate them
                GP gp_for_noise = new GP(sampledValues: noise_z, list_x: this.list_x, cov_f:cov_f,
                    lengthScale : lengthScale, sigma_f : sigma_f, sigma_jitter : sigma_jitter);

                Utility.Log("Performing Variance Regression");
                resulting_z = gp_for_noise.predict();    //Discard the variance info

                //*******************************************For Debugging
                List<string> noiseRaw = new List<string>() { ", GP, Raw" };
                foreach(var kv in resulting_z)
                {
                    //Find the z estimate against x
                    XYPair raw = noise_z.Find(z => z.x.Equals(kv.Key));
                    string y = raw==null? "" : raw.y.ToString();
                    string s = kv.Key.toString() + "," + kv.Value.mu + "," + y;
                    noiseRaw.Add(s);
                }
                fs.writeToFile(noiseRaw.ToArray());//*/



                Utility.Log("Updating Noise Term in CovMatrix");
                covMatrix.updateNoise(resulting_z.Select(kv => new XYPair(kv.Key, Math.Exp(kv.Value.mu))).ToList());    //Notice the Exp

                //3. Update GP_0

                //4. If difference < HETEROSCEDASTIC_CONVERGENCE_PERCENTAGE, converged

                //else
                counter++;
            }
            lastPredict = new Dictionary<Vector<double>, NormalDistribution>();
            list_x.ForEach(x => lastPredict.Add(x, covMatrix.getPosterior(x)));
        }

        //TODO
        public void addPoint(XYPair newPair)
        {
            lastPredict = null;
        }

        public Dictionary<Vector<double>, NormalDistribution> predict()
        {
            if (lastPredict == null)
            {
                if (heteroscedastic)
                {
                    updateInputDependentVariance();
                    return lastPredict;
                } else
                {
                    lastPredict = new Dictionary<Vector<double>, NormalDistribution>();
                    list_x.ForEach(x => lastPredict.Add(x, covMatrix.getPosterior(x)));
                }
            }
            return lastPredict;
        }
    }
}

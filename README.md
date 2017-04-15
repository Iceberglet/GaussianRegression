A Gaussian Regression Library
=============================

IMPORTANT NOTE:
For Ordinal Transformation Research Usage, Take the "ForOT" branch.
Set up dependencies in your project that needs to reference this library.
(Usually by pointing at the executable from the test build is sufficient)

Usage
-----

1. Create a Gaussian Process (GP) - See Test.cs

```csharp
var datapoints = new List<XYPair>();
//LabeledVector is contains both the design vector and its low fidelity rank. This is to help us conveniently
//get back the low fidelity information from the returned result. If not needed, just set it to 0 like below:
var pointsToPredict = new List<Vector<double>>().Select(x => new LabeledVector(0, x)).ToList();
var covarianceFunction = CovFunction.SquaredExponential(new LengthScale(5), new SigmaF(10)) + CovFunction.GaussianNoise(new SigmaJ(0.5));
var myGP = new GP(datapoints, pointsToPredict, covarianceFunction,
  heteroscedastic: true, estimateHyperPara: true
  );
//Turning hyperparameter estimate to false will use the given ones in covarianceFunction
//Turning hyperparameter estimate to true will calculate hyperparams as defined in CovFunction. E.g:
covarianceFunction.addParams(new LengthScale(5)); //This line happens in the static constructor
```

2. Obtain the posterior predictions

```csharp
Dictionary<LabeledVector, NormalDistribution> res = myGP.predict();
```

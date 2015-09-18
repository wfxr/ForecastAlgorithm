using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Statistics;
using Wfxr.DataAnalysis.Clustering;
using Wfxr.DataAnalysis.MarkovModel;
using Wfxr.Utility.ContainerExtension;

namespace ForecastAlgorithm
{
    /// <summary>
    ///  short for Fuzzy Clustering Markov Model
    /// </summary>
    public class FcmMc {
        private int[] _obsIdx;
        private readonly Fcm _fcm;
        private  ClusterReport _fcmResult;
        private readonly MarkovChain _mc;
        private int _clusterCount;
        private int _transStep;
        public Matrix<double> Center => _fcmResult?.Center;

        public FcmMc(double[,] obs) {
            _fcm = new Fcm(obs);
            _mc = new MarkovChain();
        }

        /// <summary>
        ///     模型训练
        /// </summary>
        /// <param name="clusterCount">分类数目</param>
        /// <param name="transStep">转移步数</param>
        public void Train(int clusterCount, int transStep) {
            _clusterCount = clusterCount;
            _transStep = transStep;
            _fcmResult = _fcm.Run(_clusterCount);
            _obsIdx = _fcmResult.Idx;
            _mc.Train(_obsIdx, _transStep);
        }

        public double[] Forecast() {
            // 由最后两个已知状态预测下一个最可能状态
            var obs = _obsIdx.Sub(_obsIdx.Length - _transStep, _transStep);
            var forecastIdx = _mc.Forecast(obs).Key;
            return _fcmResult.Center.Row(forecastIdx).ToArray();
        }

        public double[] WeightedForecast() {
            var obs = _obsIdx.Sub(_obsIdx.Length - _transStep, _transStep);
            var probs = _mc.T.Row(obs);
            var result = DenseVector.Create(Center.ColumnCount, 0.0);
            for (var i = 0; i < Center.RowCount; ++i) {
                var tmp = Center.Row(i).Multiply(probs[i]);
                result.Add(Center.Row(i).Multiply(probs[i]), result);
            }

            return result.ToArray();
        }
    }
}

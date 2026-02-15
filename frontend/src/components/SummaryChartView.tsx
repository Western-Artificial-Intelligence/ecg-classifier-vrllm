import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface Prediction {
  minute: number;
  probability: number;
}

interface SummaryChartViewProps {
  predictions: Prediction[];
  onMinuteClick: (minute: number) => void;
}

const SummaryChartView: React.FC<SummaryChartViewProps> = ({
  predictions,
  onMinuteClick
}) => {
  // Sort predictions by minute
  const sortedPredictions = [...predictions].sort((a, b) => a.minute - b.minute);

  // Create labels and data
  const labels = sortedPredictions.map(p => `Min ${p.minute}`);
  const data = sortedPredictions.map(p => p.probability * 100);

  // Color code based on probability
  const backgroundColors = sortedPredictions.map(p => {
    if (p.probability >= 0.75) return 'rgba(244, 67, 54, 0.8)'; // High risk - red
    if (p.probability >= 0.5) return 'rgba(255, 152, 0, 0.8)'; // Medium risk - orange
    if (p.probability >= 0.25) return 'rgba(255, 235, 59, 0.8)'; // Low-medium - yellow
    return 'rgba(76, 175, 80, 0.8)'; // Low risk - green
  });

  const borderColors = sortedPredictions.map(p => {
    if (p.probability >= 0.75) return 'rgba(244, 67, 54, 1)';
    if (p.probability >= 0.5) return 'rgba(255, 152, 0, 1)';
    if (p.probability >= 0.25) return 'rgba(255, 235, 59, 1)';
    return 'rgba(76, 175, 80, 1)';
  });

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Apnea Probability (%)',
        data,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    onClick: (_event: any, activeElements: any[]) => {
      if (activeElements.length > 0) {
        const index = activeElements[0].index;
        const minute = sortedPredictions[index].minute;
        onMinuteClick(minute);
      }
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Minute-by-Minute Apnea Probability (Click bar to view detail)',
        font: {
          size: 14,
        }
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const minute = sortedPredictions[context.dataIndex].minute;
            const prob = context.parsed.y;
            let risk = 'Low';
            if (prob >= 75) risk = 'High';
            else if (prob >= 50) risk = 'Medium';
            else if (prob >= 25) risk = 'Low-Medium';

            return [
              `Minute: ${minute}`,
              `Probability: ${prob.toFixed(1)}%`,
              `Risk Level: ${risk}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time',
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: 30,
        }
      },
      y: {
        title: {
          display: true,
          text: 'Apnea Probability (%)',
        },
        min: 0,
        max: 100,
        ticks: {
          callback: function (value: any) {
            return value + '%';
          }
        }
      },
    },
  };

  return (
    <div style={{ height: '400px', width: '100%', padding: '20px' }}>
      {predictions.length > 0 ? (
        <Bar data={chartData} options={options} />
      ) : (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#999'
        }}>
          <p>No prediction data available. Run prediction first.</p>
        </div>
      )}
    </div>
  );
};

export default SummaryChartView;

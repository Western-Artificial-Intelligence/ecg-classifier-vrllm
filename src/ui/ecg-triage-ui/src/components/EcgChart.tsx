import React, { useRef, useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface EcgChartProps {
  dataPoints: number[];
  viewWindowSize: number; // Number of data points to show at once
  startIndex: number; // Starting index of the view window
}

const EcgChart: React.FC<EcgChartProps> = ({ dataPoints, viewWindowSize, startIndex }) => {
  const chartRef = useRef<ChartJS<'line'>>(null);
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: 'ECG Signal',
        data: [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1,
        pointRadius: 0, // No points for ECG
      },
    ],
  });

  useEffect(() => {
    const endIndex = Math.min(startIndex + viewWindowSize, dataPoints.length);
    const visibleData = dataPoints.slice(startIndex, endIndex);
    const labels = Array.from({ length: visibleData.length }, (_, i) => (startIndex + i).toString());

    setChartData({
      labels: labels,
      datasets: [
        {
          ...chartData.datasets[0],
          data: visibleData,
        },
      ],
    });
  }, [dataPoints, viewWindowSize, startIndex]); // Dependencies for useEffect

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'ECG Signal',
      },
    },
    scales: {
      x: {
        type: 'category' as const, // Ensure type is correct for Chart.js v3+
        title: {
          display: true,
          text: 'Time (samples)',
        },
        grid: {
          display: false,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Amplitude',
        },
      },
    },
  };

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <Line ref={chartRef} data={chartData} options={options} />
    </div>
  );
};

export default EcgChart;

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
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  annotationPlugin
);

interface Prediction {
  minute: number;
  probability: number;
}

interface MinimapViewProps {
  dataPoints: number[];
  predictions?: Prediction[];
  currentPosition: number; // Current startIndex
  viewWindowSize: number;
  onPositionChange: (newPosition: number) => void;
}

const MinimapView: React.FC<MinimapViewProps> = ({
  dataPoints,
  predictions = [],
  currentPosition,
  viewWindowSize,
  onPositionChange
}) => {
  const chartRef = useRef<ChartJS<'line'>>(null);
  const [chartData, setChartData] = useState<{
    labels: string[];
    datasets: any[];
  }>({
    labels: [],
    datasets: [
      {
        label: 'ECG Signal (Compressed)',
        data: [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderWidth: 1,
        pointRadius: 0,
      },
    ],
  });

  // Downsample data for minimap
  const downsampleData = (data: number[], targetPoints: number = 2000): number[] => {
    if (data.length <= targetPoints) return data;

    const step = data.length / targetPoints;
    const downsampled: number[] = [];

    for (let i = 0; i < targetPoints; i++) {
      const start = Math.floor(i * step);
      const end = Math.floor((i + 1) * step);
      const chunk = data.slice(start, end);

      // Use max-min downsampling
      const max = Math.max(...chunk);
      const min = Math.min(...chunk);
      downsampled.push(max, min);
    }

    return downsampled;
  };

  useEffect(() => {
    const downsampled = downsampleData(dataPoints);
    const labels = downsampled.map((_, i) => {
      const sampleIndex = Math.floor((i / downsampled.length) * dataPoints.length);
      const minute = Math.floor(sampleIndex / 6000);
      return minute.toString();
    });

    setChartData({
      labels: labels,
      datasets: [
        {
          ...chartData.datasets[0],
          data: downsampled,
        },
      ],
    });
  }, [dataPoints]);

  // Create viewport indicator annotation
  const createViewportAnnotation = () => {
    return {
      viewport: {
        type: 'box' as const,
        xMin: Math.floor((currentPosition / dataPoints.length) * chartData.labels.length),
        xMax: Math.floor(((currentPosition + viewWindowSize) / dataPoints.length) * chartData.labels.length),
        backgroundColor: 'rgba(33, 150, 243, 0.2)',
        borderColor: 'rgb(33, 150, 243)',
        borderWidth: 2,
        label: {
          display: true,
          content: 'Current View',
          position: 'start' as const,
        }
      }
    };
  };

  // Create annotations for apneic regions
  const createApneaAnnotations = () => {
    const annotations: any = {};
    const SAMPLES_PER_MINUTE = 6000;
    const APNEA_THRESHOLD = 0.5;

    predictions.forEach((pred) => {
      if (pred.probability >= APNEA_THRESHOLD) {
        const minuteStartSample = pred.minute * SAMPLES_PER_MINUTE;
        const minuteEndSample = (pred.minute + 1) * SAMPLES_PER_MINUTE;

        const xMin = Math.floor((minuteStartSample / dataPoints.length) * chartData.labels.length);
        const xMax = Math.floor((minuteEndSample / dataPoints.length) * chartData.labels.length);

        annotations[`apnea_${pred.minute}`] = {
          type: 'box' as const,
          xMin,
          xMax,
          backgroundColor: `rgba(255, 99, 132, ${pred.probability * 0.3})`,
          borderColor: 'rgba(255, 99, 132, 0.8)',
          borderWidth: 1,
        };
      }
    });

    return annotations;
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    onClick: (_event: any, _activeElements: any[], chart: any) => {
      // Get click position
      const canvasPosition = chart.canvas.getBoundingClientRect();
      const _x = _event.clientX - canvasPosition.left;
      // Convert pixel position to data index
      const chartArea = chart.chartArea;
      const chartWidth = chartArea.right - chartArea.left;
      const clickPercent = (_x - chartArea.left) / chartWidth;

      if (clickPercent >= 0 && clickPercent <= 1) {
        const newPosition = Math.floor(clickPercent * dataPoints.length);
        // Center the view on the clicked position
        const centeredPosition = Math.max(0, Math.min(
          newPosition - Math.floor(viewWindowSize / 2),
          dataPoints.length - viewWindowSize
        ));
        onPositionChange(centeredPosition);
      }
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Record Overview (Click to navigate)',
      },
      annotation: {
        annotations: {
          ...createApneaAnnotations(),
          ...createViewportAnnotation()
        }
      },
      tooltip: {
        enabled: false, // Disable tooltips for minimap
      }
    },
    scales: {
      x: {
        type: 'category' as const,
        title: {
          display: true,
          text: 'Time (minutes)',
        },
        ticks: {
          maxTicksLimit: 20, // Limit number of labels
        }
      },
      y: {
        title: {
          display: false,
        },
        ticks: {
          display: false, // Hide y-axis labels
        }
      },
    },
  };

  return (
    <div style={{ height: '200px', width: '100%' }}>
      <Line ref={chartRef} data={chartData} options={options} />
    </div>
  );
};

export default MinimapView;

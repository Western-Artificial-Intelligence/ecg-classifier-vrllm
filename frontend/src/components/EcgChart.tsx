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

interface EcgChartProps {
  dataPoints: number[];
  viewWindowSize: number; // Number of data points to show at once
  startIndex: number; // Starting index of the view window
  predictions?: Prediction[];
  onSegmentClick?: (minute: number) => void;
  isFullView?: boolean; // Whether displaying full record
  showAnnotations?: boolean; // Whether to show analysis overlays
}

const EcgChart: React.FC<EcgChartProps> = ({ 
  dataPoints, 
  viewWindowSize, 
  startIndex, 
  predictions = [],
  onSegmentClick,
  isFullView = false,
  showAnnotations = true
}) => {
  // Downsample data for full view
  const downsampleData = (data: number[], targetPoints: number = 3000): number[] => {
    if (data.length <= targetPoints) return data;
    
    const step = data.length / targetPoints;
    const downsampled: number[] = [];
    
    for (let i = 0; i < targetPoints; i++) {
      const start = Math.floor(i * step);
      const end = Math.floor((i + 1) * step);
      const chunk = data.slice(start, end);
      
      // Use max-min downsampling for better representation
      const max = Math.max(...chunk);
      const min = Math.min(...chunk);
      downsampled.push(max, min);
    }
    
    return downsampled;
  };
  const chartRef = useRef<ChartJS<'line'>>(null);
  const [chartData, setChartData] = useState<{
    labels: string[];
    datasets: Array<{
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
      tension: number;
      pointRadius: number;
    }>;
  }>({
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
    // Determine if we should downsample
    const shouldDownsample = isFullView && dataPoints.length > 5000;
    
    const endIndex = Math.min(startIndex + viewWindowSize, dataPoints.length);
    let visibleData = dataPoints.slice(startIndex, endIndex);
    
    // Downsample if needed
    if (shouldDownsample) {
      visibleData = downsampleData(visibleData);
    }
    
    // Create labels - for full view, use minute markers
    let labels: string[];
    if (isFullView) {
      // Assume 100 Hz sampling rate, so 6000 samples = 1 minute
      labels = visibleData.map((_, i) => {
        const sampleIndex = Math.floor((i / visibleData.length) * dataPoints.length);
        return Math.floor(sampleIndex / 6000).toString();
      });
    } else {
      labels = Array.from({ length: visibleData.length }, (_, i) => ((startIndex + i) / 100).toFixed(2));
    }

    setChartData({
      labels: labels,
      datasets: [
        {
          label: 'ECG Signal',
          data: visibleData,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1,
          pointRadius: 0,
        },
      ],
    });
  }, [dataPoints, viewWindowSize, startIndex, isFullView]); // Dependencies for useEffect

  // Create annotations for apneic regions
  const createAnnotations = () => {
    const annotations: any = {};
    const SAMPLES_PER_MINUTE = 6000; // 100Hz * 60 seconds
    const APNEA_THRESHOLD = 0.5;

    predictions
      .filter(p => p.probability >= APNEA_THRESHOLD)
      .forEach((pred, index) => {
        const minuteStartSample = pred.minute * SAMPLES_PER_MINUTE;
        const minuteEndSample = (pred.minute + 1) * SAMPLES_PER_MINUTE;

        // Only show annotation if it's visible in the current window
        if (minuteEndSample >= startIndex && minuteStartSample < startIndex + viewWindowSize) {
          const visibleStart = Math.max(minuteStartSample, startIndex) - startIndex;
          const visibleEnd = Math.min(minuteEndSample, startIndex + viewWindowSize) - startIndex;

          annotations[`apnea_${index}`] = {
            type: 'box',
            xMin: visibleStart,
            xMax: visibleEnd,
            backgroundColor: pred.probability >= 0.7 
              ? 'rgba(255, 99, 71, 0.2)' // High confidence: red
              : 'rgba(255, 165, 0, 0.2)', // Medium confidence: orange
            borderColor: pred.probability >= 0.7 
              ? 'rgba(255, 99, 71, 0.5)'
              : 'rgba(255, 165, 0, 0.5)',
            borderWidth: 1,
            label: {
              display: true,
              content: `Apnea (${(pred.probability * 100).toFixed(0)}%)`,
              position: 'start',
              color: '#333',
              font: {
                size: 10
              }
            },
            click: () => {
              if (onSegmentClick) {
                onSegmentClick(pred.minute);
              }
            }
          };
        }
      });

    return annotations;
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    onClick: (event: any, _elements: any[], chart: any) => {
      // Check if click is on an annotation
      if (chart && chart.options && chart.options.plugins && chart.options.plugins.annotation) {
        const annotations = chart.options.plugins.annotation.annotations;
        if (annotations) {
          // Find which annotation was clicked
          const x = event.x;
          
          for (const key in annotations) {
            const annotation = annotations[key];
            if (annotation.type === 'box') {
              // Calculate pixel positions
              const xScale = chart.scales.x;
              const xMin = xScale.getPixelForValue(annotation.xMin);
              const xMax = xScale.getPixelForValue(annotation.xMax);
              
              if (x >= xMin && x <= xMax) {
                if (annotation.click) {
                  annotation.click();
                }
                break;
              }
            }
          }
        }
      }
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'ECG Signal',
      },
      annotation: {
        annotations: showAnnotations ? createAnnotations() : {}
      },
      tooltip: {
        enabled: !isFullView, // Disable tooltips for full view (performance)
        callbacks: {
          afterLabel: (context: any) => {
            if (isFullView) return '';
            const sampleIndex = startIndex + context.dataIndex;
            const minute = Math.floor(sampleIndex / 6000);
            const prediction = predictions.find(p => p.minute === minute);
            if (prediction) {
              return `Minute ${minute}: Apnea probability ${(prediction.probability * 100).toFixed(1)}%`;
            }
            return '';
          }
        }
      }
    },
    scales: {
      x: {
        type: 'category' as const,
        title: {
          display: true,
          text: isFullView ? 'Time (minutes)' : 'Time (s)',
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

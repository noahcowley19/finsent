'use client';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
  ChartData,
} from 'chart.js';
import { Line, Bar, Doughnut, Pie } from 'react-chartjs-2';
import { cn } from '@/lib/utils';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const colors = {
  positive: '#10b981',
  negative: '#ef4444',
  neutral: '#6b7280',
  warning: '#f59e0b',
  primary: '#1e293b',
  secondary: '#64748b',
  blue: '#3b82f6',
  purple: '#8b5cf6',
  cyan: '#06b6d4',
};

interface ChartWrapperProps {
  children: React.ReactNode;
  className?: string;
}

function ChartWrapper({ children, className }: ChartWrapperProps) {
  return (
    <div className={cn('chart-container', className)}>
      {children}
    </div>
  );
}

interface LineChartProps {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    color?: string;
    fill?: boolean;
  }>;
  title?: string;
  className?: string;
  yAxisLabel?: string;
  showLegend?: boolean;
}

export function LineChart({ 
  labels, 
  datasets, 
  title,
  className,
  yAxisLabel,
  showLegend = true
}: LineChartProps) {
  const data: ChartData<'line'> = {
    labels,
    datasets: datasets.map((ds, index) => ({
      label: ds.label,
      data: ds.data,
      borderColor: ds.color || Object.values(colors)[index % Object.values(colors).length],
      backgroundColor: ds.fill 
        ? `${ds.color || Object.values(colors)[index % Object.values(colors).length]}20`
        : 'transparent',
      fill: ds.fill || false,
      tension: 0.3,
      pointRadius: 2,
      pointHoverRadius: 5,
    })),
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: showLegend, position: 'top' },
      title: { display: !!title, text: title },
    },
    scales: {
      y: {
        title: { display: !!yAxisLabel, text: yAxisLabel },
        grid: { color: '#e2e8f0' },
      },
      x: {
        grid: { display: false },
      },
    },
  };

  return (
    <ChartWrapper className={className}>
      <Line data={data} options={options} />
    </ChartWrapper>
  );
}

interface BarChartProps {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    color?: string;
  }>;
  title?: string;
  className?: string;
  horizontal?: boolean;
  stacked?: boolean;
}

export function BarChart({ 
  labels, 
  datasets, 
  title,
  className,
  horizontal = false,
  stacked = false
}: BarChartProps) {
  const data: ChartData<'bar'> = {
    labels,
    datasets: datasets.map((ds, index) => ({
      label: ds.label,
      data: ds.data,
      backgroundColor: ds.color || Object.values(colors)[index % Object.values(colors).length],
      borderRadius: 4,
    })),
  };

  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? 'y' : 'x',
    plugins: {
      legend: { display: datasets.length > 1, position: 'top' },
      title: { display: !!title, text: title },
    },
    scales: {
      y: {
        stacked,
        grid: { color: '#e2e8f0' },
      },
      x: {
        stacked,
        grid: { display: false },
      },
    },
  };

  return (
    <ChartWrapper className={className}>
      <Bar data={data} options={options} />
    </ChartWrapper>
  );
}

interface DoughnutChartProps {
  labels: string[];
  data: number[];
  colors?: string[];
  title?: string;
  className?: string;
  showLegend?: boolean;
}

export function DoughnutChart({ 
  labels, 
  data, 
  colors: customColors,
  title,
  className,
  showLegend = true
}: DoughnutChartProps) {
  const defaultColors = ['#10b981', '#ef4444', '#6b7280', '#f59e0b', '#3b82f6', '#8b5cf6'];
  
  const chartData: ChartData<'doughnut'> = {
    labels,
    datasets: [{
      data,
      backgroundColor: customColors || defaultColors.slice(0, data.length),
      borderWidth: 0,
    }],
  };

  const options: ChartOptions<'doughnut'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: showLegend, position: 'right' },
      title: { display: !!title, text: title },
    },
    cutout: '60%',
  };

  return (
    <ChartWrapper className={className}>
      <Doughnut data={chartData} options={options} />
    </ChartWrapper>
  );
}

interface PieChartProps {
  labels: string[];
  data: number[];
  colors?: string[];
  title?: string;
  className?: string;
}

export function PieChart({ 
  labels, 
  data, 
  colors: customColors,
  title,
  className
}: PieChartProps) {
  const defaultColors = ['#10b981', '#ef4444', '#6b7280', '#f59e0b', '#3b82f6', '#8b5cf6'];
  
  const chartData: ChartData<'pie'> = {
    labels,
    datasets: [{
      data,
      backgroundColor: customColors || defaultColors.slice(0, data.length),
      borderWidth: 0,
    }],
  };

  const options: ChartOptions<'pie'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'right' },
      title: { display: !!title, text: title },
    },
  };

  return (
    <ChartWrapper className={className}>
      <Pie data={chartData} options={options} />
    </ChartWrapper>
  );
}

interface PriceChartProps {
  dates: string[];
  prices: number[];
  volumes?: number[];
  title?: string;
  className?: string;
}

export function PriceChart({ 
  dates, 
  prices, 
  volumes,
  title,
  className
}: PriceChartProps) {
  const priceChange = prices.length > 1 ? prices[prices.length - 1] - prices[0] : 0;
  const lineColor = priceChange >= 0 ? colors.positive : colors.negative;

  const data: ChartData<'line'> = {
    labels: dates,
    datasets: [{
      label: 'Price',
      data: prices,
      borderColor: lineColor,
      backgroundColor: `${lineColor}20`,
      fill: true,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 5,
    }],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index',
    },
    plugins: {
      legend: { display: false },
      title: { display: !!title, text: title },
      tooltip: {
  callbacks: {
    label: (ctx) => `$${ctx.parsed.y?.toFixed(2) ?? 'N/A'}`,
  },
},
    },
    scales: {
      y: {
        grid: { color: '#e2e8f0' },
        ticks: {
          callback: (value) => `$${value}`,
        },
      },
      x: {
        grid: { display: false },
        ticks: {
          maxTicksLimit: 6,
        },
      },
    },
  };

  return (
    <ChartWrapper className={className}>
      <Line data={data} options={options} />
    </ChartWrapper>
  );
}

export function SentimentChart({
  positive,
  negative,
  neutral,
  className
}: {
  positive: number;
  negative: number;
  neutral: number;
  className?: string;
}) {
  return (
    <DoughnutChart
      labels={['Positive', 'Negative', 'Neutral']}
      data={[positive, negative, neutral]}
      colors={[colors.positive, colors.negative, colors.neutral]}
      className={className}
    />
  );
}

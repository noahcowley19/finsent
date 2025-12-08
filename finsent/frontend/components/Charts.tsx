'use client';

import { useEffect, useRef } from 'react';

interface PriceChartProps {
  dates: string[];
  prices: number[];
  height?: number;
  showVolume?: boolean;
  volumes?: number[];
}

export function PriceChart({ dates, prices, height = 300 }: PriceChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || prices.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get device pixel ratio for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const chartHeight = rect.height;
    const padding = { top: 20, right: 60, bottom: 40, left: 10 };
    const chartWidth = width - padding.left - padding.right;
    const drawHeight = chartHeight - padding.top - padding.bottom;

    // Calculate min/max with padding
    const validPrices = prices.filter(p => p !== null && !isNaN(p));
    const minPrice = Math.min(...validPrices);
    const maxPrice = Math.max(...validPrices);
    const priceRange = maxPrice - minPrice || 1;
    const paddedMin = minPrice - priceRange * 0.05;
    const paddedMax = maxPrice + priceRange * 0.05;
    const paddedRange = paddedMax - paddedMin;

    // Determine color based on trend
    const startPrice = validPrices[0];
    const endPrice = validPrices[validPrices.length - 1];
    const isPositive = endPrice >= startPrice;
    const lineColor = isPositive ? '#00e5a0' : '#ff6b6b';
    const gradientTop = isPositive ? 'rgba(0, 229, 160, 0.3)' : 'rgba(255, 107, 107, 0.3)';
    const gradientBottom = isPositive ? 'rgba(0, 229, 160, 0)' : 'rgba(255, 107, 107, 0)';

    // Clear canvas
    ctx.clearRect(0, 0, width, chartHeight);

    // Create gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, chartHeight - padding.bottom);
    gradient.addColorStop(0, gradientTop);
    gradient.addColorStop(1, gradientBottom);

    // Calculate points
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < prices.length; i++) {
      if (prices[i] !== null && !isNaN(prices[i])) {
        const x = padding.left + (i / (prices.length - 1)) * chartWidth;
        const y = padding.top + (1 - (prices[i] - paddedMin) / paddedRange) * drawHeight;
        points.push({ x, y });
      }
    }

    if (points.length < 2) return;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (i / 4) * drawHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw filled area
    ctx.beginPath();
    ctx.moveTo(points[0].x, chartHeight - padding.bottom);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, chartHeight - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Draw end point
    const lastPoint = points[points.length - 1];
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = lineColor;
    ctx.fill();
    
    // Glow effect
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = `${lineColor}33`;
    ctx.fill();

    // Y-axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 4; i++) {
      const price = paddedMax - (i / 4) * paddedRange;
      const y = padding.top + (i / 4) * drawHeight;
      ctx.fillText(`$${price.toFixed(2)}`, width - 8, y + 4);
    }

    // X-axis labels (show 5 labels)
    ctx.textAlign = 'center';
    const labelIndices = [0, Math.floor(dates.length * 0.25), Math.floor(dates.length * 0.5), Math.floor(dates.length * 0.75), dates.length - 1];
    labelIndices.forEach(idx => {
      if (dates[idx]) {
        const x = padding.left + (idx / (dates.length - 1)) * chartWidth;
        const dateStr = dates[idx].split(' ')[0]; // Just date, no time
        ctx.fillText(dateStr, x, chartHeight - 12);
      }
    });

  }, [dates, prices]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: `${height}px`,
        display: 'block',
      }}
    />
  );
}

interface BarChartProps {
  labels: string[];
  data: number[];
  colors?: string[];
  height?: number;
}

export function BarChart({ labels, data, colors, height = 200 }: BarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const chartHeight = rect.height;
    const padding = { top: 20, right: 20, bottom: 50, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const drawHeight = chartHeight - padding.top - padding.bottom;

    const maxValue = Math.max(...data.map(Math.abs));
    const hasNegative = data.some(d => d < 0);
    const minValue = hasNegative ? -maxValue : 0;
    const range = maxValue - minValue || 1;

    // Clear
    ctx.clearRect(0, 0, width, chartHeight);

    // Zero line
    const zeroY = hasNegative 
      ? padding.top + (maxValue / range) * drawHeight
      : chartHeight - padding.bottom;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(width - padding.right, zeroY);
    ctx.stroke();

    // Draw bars
    const barWidth = (chartWidth / data.length) * 0.7;
    const gap = (chartWidth / data.length) * 0.3;

    data.forEach((value, i) => {
      const x = padding.left + (i * chartWidth / data.length) + gap / 2;
      const barHeight = (Math.abs(value) / range) * drawHeight;
      const y = value >= 0 ? zeroY - barHeight : zeroY;

      const color = colors?.[i] || (value >= 0 ? '#00e5a0' : '#ff6b6b');
      
      // Draw bar with rounded top
      ctx.beginPath();
      const radius = Math.min(4, barWidth / 2);
      if (value >= 0) {
        ctx.moveTo(x, y + barHeight);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.lineTo(x + barWidth - radius, y);
        ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + radius);
        ctx.lineTo(x + barWidth, y + barHeight);
      } else {
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + barHeight - radius);
        ctx.quadraticCurveTo(x, y + barHeight, x + radius, y + barHeight);
        ctx.lineTo(x + barWidth - radius, y + barHeight);
        ctx.quadraticCurveTo(x + barWidth, y + barHeight, x + barWidth, y + barHeight - radius);
        ctx.lineTo(x + barWidth, y);
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    });

    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    
    labels.forEach((label, i) => {
      const x = padding.left + (i * chartWidth / data.length) + (chartWidth / data.length) / 2;
      ctx.fillText(label, x, chartHeight - 12);
    });

  }, [labels, data, colors]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: `${height}px`,
        display: 'block',
      }}
    />
  );
}

interface DonutChartProps {
  data: { label: string; value: number; color?: string }[];
  size?: number;
  thickness?: number;
  centerLabel?: string;
  centerValue?: string;
}

export function DonutChart({ 
  data, 
  size = 200, 
  thickness = 30,
  centerLabel,
  centerValue 
}: DonutChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const defaultColors = [
    '#00e5a0', '#00a3ff', '#a855f7', '#f59e0b', 
    '#ef4444', '#06b6d4', '#84cc16', '#f472b6'
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const centerX = size / 2;
    const centerY = size / 2;
    const radius = (size - thickness) / 2;

    const total = data.reduce((sum, d) => sum + d.value, 0);
    let currentAngle = -Math.PI / 2;

    // Clear
    ctx.clearRect(0, 0, size, size);

    // Draw segments
    data.forEach((segment, i) => {
      const sliceAngle = (segment.value / total) * Math.PI * 2;
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      ctx.arc(centerX, centerY, radius - thickness, currentAngle + sliceAngle, currentAngle, true);
      ctx.closePath();
      
      ctx.fillStyle = segment.color || defaultColors[i % defaultColors.length];
      ctx.fill();
      
      currentAngle += sliceAngle;
    });

    // Center text
    if (centerValue) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.font = 'bold 24px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(centerValue, centerX, centerY - 8);
    }
    
    if (centerLabel) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '12px system-ui';
      ctx.fillText(centerLabel, centerX, centerY + 16);
    }

  }, [data, size, thickness, centerLabel, centerValue]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${size}px`,
        height: `${size}px`,
        display: 'block',
      }}
    />
  );
}

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showDot?: boolean;
}

// Alias exports for compatibility
export { PriceChart as LineChart };
export { DonutChart as DoughnutChart };
export { DonutChart as PieChart };
export { BarChart as SentimentChart };

export function Sparkline({ 
  data, 
  width = 80, 
  height = 24, 
  color,
  showDot = true 
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const padding = 2;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const isPositive = data[data.length - 1] >= data[0];
    const lineColor = color || (isPositive ? '#00e5a0' : '#ff6b6b');

    ctx.clearRect(0, 0, width, height);

    // Draw line
    ctx.beginPath();
    data.forEach((value, i) => {
      const x = padding + (i / (data.length - 1)) * chartWidth;
      const y = padding + (1 - (value - min) / range) * chartHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // End dot
    if (showDot) {
      const lastX = width - padding;
      const lastY = padding + (1 - (data[data.length - 1] - min) / range) * chartHeight;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 2, 0, Math.PI * 2);
      ctx.fillStyle = lineColor;
      ctx.fill();
    }

  }, [data, width, height, color, showDot]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        display: 'block',
      }}
    />
  );
}

'use client';

import { useEffect, useRef, useState } from 'react';

interface ChartDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AdvancedChartProps {
  data: ChartDataPoint[];
  type: 'line' | 'candlestick';
  showVolume?: boolean;
  showMA?: boolean;
  ma20?: number[];
  ma50?: number[];
  ma200?: number[];
  height?: number;
  onTypeChange?: (type: 'line' | 'candlestick') => void;
}

export default function AdvancedChart({
  data,
  type = 'line',
  showVolume = true,
  showMA = false,
  ma20 = [],
  ma50 = [],
  ma200 = [],
  height = 400,
}: AdvancedChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);

  useEffect(() => {
    if (!data || data.length === 0) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const chartHeight = showVolume ? rect.height * 0.7 : rect.height;
    const volumeHeight = showVolume ? rect.height * 0.25 : 0;
    const padding = { top: 30, right: 70, bottom: showVolume ? 80 : 50, left: 10 };
    const chartWidth = width - padding.left - padding.right;
    const drawHeight = chartHeight - padding.top - padding.bottom;
    const volumeDrawHeight = volumeHeight - 10;

    // Clear canvas
    ctx.clearRect(0, 0, width, rect.height);

    // Calculate visible data range based on zoom and pan
    const visibleDataCount = Math.max(Math.floor(data.length / zoomLevel), 10);
    const startIdx = Math.max(0, Math.min(data.length - visibleDataCount, Math.floor(panOffset)));
    const endIdx = Math.min(data.length, startIdx + visibleDataCount);
    const visibleData = data.slice(startIdx, endIdx);

    if (visibleData.length === 0) return;

    // Calculate price range
    const prices = visibleData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice || 1;
    const paddedMin = minPrice - priceRange * 0.05;
    const paddedMax = maxPrice + priceRange * 0.05;
    const paddedRange = paddedMax - paddedMin;

    // Calculate volume range
    const maxVolume = Math.max(...visibleData.map(d => d.volume));

    // Helper functions
    const getX = (index: number) => padding.left + (index / (visibleData.length - 1)) * chartWidth;
    const getY = (price: number) => padding.top + (1 - (price - paddedMin) / paddedRange) * drawHeight;
    const getVolumeY = (volume: number) => chartHeight + 10 + (1 - volume / maxVolume) * volumeDrawHeight;

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * drawHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Vertical grid lines
    const gridLineCount = Math.min(8, visibleData.length);
    for (let i = 0; i <= gridLineCount; i++) {
      const x = padding.left + (i / gridLineCount) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, chartHeight - padding.bottom);
      ctx.stroke();
    }

    // Draw chart based on type
    if (type === 'candlestick') {
      // Draw candlesticks
      const candleWidth = Math.max(2, chartWidth / visibleData.length * 0.7);
      
      visibleData.forEach((point, i) => {
        const x = getX(i);
        const openY = getY(point.open);
        const closeY = getY(point.close);
        const highY = getY(point.high);
        const lowY = getY(point.low);
        
        const isUp = point.close >= point.open;
        const bodyColor = isUp ? '#00e5a0' : '#ff6b6b';
        const bodyAlpha = hoveredIndex === i + startIdx ? 1 : 0.9;

        // Draw wick
        ctx.strokeStyle = bodyColor;
        ctx.globalAlpha = bodyAlpha;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        // Draw body
        const bodyTop = Math.min(openY, closeY);
        const bodyHeight = Math.abs(closeY - openY) || 1;
        
        ctx.fillStyle = bodyColor;
        ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
        
        // Highlight on hover
        if (hoveredIndex === i + startIdx) {
          ctx.strokeStyle = bodyColor;
          ctx.lineWidth = 2;
          ctx.strokeRect(x - candleWidth / 2 - 2, bodyTop - 2, candleWidth + 4, bodyHeight + 4);
        }
        
        ctx.globalAlpha = 1;
      });
    } else {
      // Draw line chart
      // Draw gradient fill
      const gradient = ctx.createLinearGradient(0, padding.top, 0, chartHeight - padding.bottom);
      const isPositive = visibleData[visibleData.length - 1].close >= visibleData[0].close;
      const lineColor = isPositive ? '#00e5a0' : '#ff6b6b';
      
      gradient.addColorStop(0, isPositive ? 'rgba(0, 229, 160, 0.25)' : 'rgba(255, 107, 107, 0.25)');
      gradient.addColorStop(1, isPositive ? 'rgba(0, 229, 160, 0)' : 'rgba(255, 107, 107, 0)');

      ctx.beginPath();
      ctx.moveTo(getX(0), chartHeight - padding.bottom);
      visibleData.forEach((point, i) => {
        ctx.lineTo(getX(i), getY(point.close));
      });
      ctx.lineTo(getX(visibleData.length - 1), chartHeight - padding.bottom);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      // Draw line
      ctx.beginPath();
      visibleData.forEach((point, i) => {
        const x = getX(i);
        const y = getY(point.close);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Draw points on hover
      if (hoveredIndex !== null && hoveredIndex >= startIdx && hoveredIndex < endIdx) {
        const i = hoveredIndex - startIdx;
        const x = getX(i);
        const y = getY(visibleData[i].close);
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = lineColor;
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = `${lineColor}44`;
        ctx.fill();
      }
    }

    // Draw moving averages
    if (showMA) {
      const drawMA = (maData: number[], color: string, lineWidth: number) => {
        if (maData.length === 0) return;
        
        const visibleMA = maData.slice(startIdx, endIdx);
        ctx.beginPath();
        ctx.globalAlpha = 0.7;
        visibleMA.forEach((value, i) => {
          if (value && !isNaN(value)) {
            const x = getX(i);
            const y = getY(value);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        ctx.globalAlpha = 1;
      };

      drawMA(ma20, '#fbbf24', 1.5);
      drawMA(ma50, '#3b82f6', 1.5);
      drawMA(ma200, '#a855f7', 1.5);
    }

    // Draw volume bars
    if (showVolume) {
      visibleData.forEach((point, i) => {
        const x = getX(i);
        const volumeY = getVolumeY(point.volume);
        const barHeight = chartHeight + 10 + volumeDrawHeight - volumeY;
        const barWidth = Math.max(1, chartWidth / visibleData.length * 0.7);
        
        const isUp = point.close >= point.open;
        ctx.fillStyle = isUp ? 'rgba(0, 229, 160, 0.4)' : 'rgba(255, 107, 107, 0.4)';
        ctx.fillRect(x - barWidth / 2, volumeY, barWidth, barHeight);
      });
    }

    // Y-axis labels (price)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = paddedMax - (i / 5) * paddedRange;
      const y = padding.top + (i / 5) * drawHeight;
      ctx.fillText(`$${price.toFixed(2)}`, width - 8, y + 4);
    }

    // X-axis labels (dates)
    ctx.textAlign = 'center';
    const labelIndices = [0, Math.floor(visibleData.length * 0.25), Math.floor(visibleData.length * 0.5), Math.floor(visibleData.length * 0.75), visibleData.length - 1];
    labelIndices.forEach(idx => {
      if (visibleData[idx]) {
        const x = getX(idx);
        const dateStr = new Date(visibleData[idx].date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        ctx.fillText(dateStr, x, chartHeight - padding.bottom + 20);
      }
    });

    // Volume label
    if (showVolume) {
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText('Volume', padding.left + 4, chartHeight + 20);
    }

  }, [data, type, showVolume, showMA, ma20, ma50, ma200, hoveredIndex, zoomLevel, panOffset]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const visibleDataCount = Math.max(Math.floor(data.length / zoomLevel), 10);
    const startIdx = Math.max(0, Math.min(data.length - visibleDataCount, Math.floor(panOffset)));
    const endIdx = Math.min(data.length, startIdx + visibleDataCount);

    const padding = { left: 10, right: 70 };
    const chartWidth = rect.width - padding.left - padding.right;
    
    if (x >= padding.left && x <= rect.width - padding.right) {
      const relativeX = x - padding.left;
      const index = Math.floor((relativeX / chartWidth) * (endIdx - startIdx)) + startIdx;
      
      if (index >= 0 && index < data.length) {
        setHoveredIndex(index);
        setTooltipPos({ x: e.clientX, y: e.clientY });
      }
    } else {
      setHoveredIndex(null);
    }

    if (isDragging) {
      const delta = (e.clientX - dragStart) / chartWidth * (endIdx - startIdx);
      setPanOffset(Math.max(0, Math.min(data.length - visibleDataCount, panOffset - delta)));
      setDragStart(e.clientX);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart(e.clientX);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoomLevel(Math.max(1, Math.min(5, zoomLevel * delta)));
  };

  const tooltipData = hoveredIndex !== null && data[hoveredIndex] ? data[hoveredIndex] : null;

  return (
    <div style={{ position: 'relative', width: '100%', height: `${height}px` }}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setHoveredIndex(null); setIsDragging(false); }}
        onWheel={handleWheel}
        style={{
          width: '100%',
          height: '100%',
          cursor: isDragging ? 'grabbing' : 'crosshair',
        }}
      />
      
      {/* Tooltip */}
      {tooltipData && hoveredIndex !== null && (
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x + 20,
            top: tooltipPos.y - 80,
            background: 'rgba(10, 14, 23, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '12px 16px',
            pointerEvents: 'none',
            zIndex: 1000,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
            minWidth: '180px',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>
            {new Date(tooltipData.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 12px', fontSize: '12px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Open:</span>
            <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', textAlign: 'right' }}>${tooltipData.open.toFixed(2)}</span>
            
            <span style={{ color: 'var(--text-secondary)' }}>High:</span>
            <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', textAlign: 'right', color: 'var(--positive)' }}>${tooltipData.high.toFixed(2)}</span>
            
            <span style={{ color: 'var(--text-secondary)' }}>Low:</span>
            <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', textAlign: 'right', color: 'var(--negative)' }}>${tooltipData.low.toFixed(2)}</span>
            
            <span style={{ color: 'var(--text-secondary)' }}>Close:</span>
            <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', textAlign: 'right' }}>${tooltipData.close.toFixed(2)}</span>
            
            <span style={{ color: 'var(--text-secondary)' }}>Volume:</span>
            <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', textAlign: 'right', fontSize: '11px' }}>
              {(tooltipData.volume / 1000000).toFixed(2)}M
            </span>
          </div>
        </div>
      )}

      {/* Zoom indicator */}
      {zoomLevel > 1 && (
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(0, 212, 170, 0.15)',
            border: '1px solid rgba(0, 212, 170, 0.3)',
            borderRadius: '8px',
            padding: '6px 12px',
            fontSize: '11px',
            fontWeight: 600,
            color: 'var(--accent)',
          }}
        >
          Zoom: {zoomLevel.toFixed(1)}x
        </div>
      )}
    </div>
  );
}

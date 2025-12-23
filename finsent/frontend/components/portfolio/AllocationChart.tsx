'use client';

// =============================================================================
// ALLOCATION CHART COMPONENT
// =============================================================================
// Donut chart showing portfolio allocation
//
// Location: frontend/components/portfolio/AllocationChart.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface AllocationItem {
  label: string;
  value: number;
  color: string;
}

export interface AllocationChartProps {
  data: AllocationItem[];
  title?: string;
  isLoading?: boolean;
}

const defaultColors = [
  '#131D4F', // navy
  '#954C2E', // terra
  '#22C55E', // success
  '#F59E0B', // warning
  '#6366F1', // indigo
  '#EC4899', // pink
  '#14B8A6', // teal
  '#8B5CF6', // violet
];

export const AllocationChart: React.FC<AllocationChartProps> = ({
  data,
  title = 'Allocation',
  isLoading = false,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-24 h-6 rounded bg-cream-100 animate-pulse mb-6" />
        <div className="flex items-center gap-8">
          <div className="w-40 h-40 rounded-full bg-cream-100 animate-pulse" />
          <div className="flex-1 space-y-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="flex items-center gap-3 animate-pulse">
                <div className="w-3 h-3 rounded-full bg-cream-100" />
                <div className="w-24 h-4 rounded bg-cream-100" />
                <div className="w-12 h-4 rounded bg-cream-100 ml-auto" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
          {title}
        </h3>
        <div className="h-40 flex items-center justify-center text-neutral-500">
          No allocation data available
        </div>
      </div>
    );
  }

  // Calculate total and percentages
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const dataWithPercent = data.map((item, index) => ({
    ...item,
    percent: (item.value / total) * 100,
    color: item.color || defaultColors[index % defaultColors.length],
  }));

  // Generate donut chart paths
  const generateDonutPath = (startAngle: number, endAngle: number, outerRadius: number, innerRadius: number) => {
    const startOuter = polarToCartesian(50, 50, outerRadius, startAngle);
    const endOuter = polarToCartesian(50, 50, outerRadius, endAngle);
    const startInner = polarToCartesian(50, 50, innerRadius, endAngle);
    const endInner = polarToCartesian(50, 50, innerRadius, startAngle);

    const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;

    return [
      `M ${startOuter.x} ${startOuter.y}`,
      `A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 1 ${endOuter.x} ${endOuter.y}`,
      `L ${startInner.x} ${startInner.y}`,
      `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${endInner.x} ${endInner.y}`,
      'Z',
    ].join(' ');
  };

  const polarToCartesian = (cx: number, cy: number, radius: number, angle: number) => {
    const rad = ((angle - 90) * Math.PI) / 180;
    return {
      x: cx + radius * Math.cos(rad),
      y: cy + radius * Math.sin(rad),
    };
  };

  // Generate segments
  let currentAngle = 0;
  const segments = dataWithPercent.map((item, index) => {
    const startAngle = currentAngle;
    const sweepAngle = (item.percent / 100) * 360;
    currentAngle += sweepAngle;

    return {
      ...item,
      path: generateDonutPath(startAngle, startAngle + sweepAngle - 0.5, 45, 28),
      index,
    };
  });

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
        {title}
      </h3>

      <div className="flex flex-col sm:flex-row items-center gap-8">
        {/* Chart */}
        <div className="relative w-40 h-40 flex-shrink-0">
          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
            {segments.map((segment) => (
              <path
                key={segment.index}
                d={segment.path}
                fill={segment.color}
                className="transition-opacity duration-fast cursor-pointer"
                opacity={hoveredIndex === null || hoveredIndex === segment.index ? 1 : 0.3}
                onMouseEnter={() => setHoveredIndex(segment.index)}
                onMouseLeave={() => setHoveredIndex(null)}
              />
            ))}
          </svg>

          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            {hoveredIndex !== null ? (
              <>
                <p className="text-body-sm font-semibold text-navy-900">
                  {dataWithPercent[hoveredIndex].percent.toFixed(1)}%
                </p>
                <p className="text-caption text-neutral-500 text-center px-2 truncate max-w-[80px]">
                  {dataWithPercent[hoveredIndex].label}
                </p>
              </>
            ) : (
              <>
                <p className="text-body-sm font-semibold text-navy-900">
                  {data.length}
                </p>
                <p className="text-caption text-neutral-500">
                  Holdings
                </p>
              </>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="flex-1 w-full space-y-2">
          {dataWithPercent.slice(0, 6).map((item, index) => (
            <div
              key={index}
              className={`
                flex items-center gap-3 py-1.5 px-2 -mx-2 rounded-lg cursor-pointer
                transition-colors duration-fast
                ${hoveredIndex === index ? 'bg-cream-50' : 'hover:bg-cream-50'}
              `}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <span
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-body-sm text-navy-900 truncate flex-1">
                {item.label}
              </span>
              <span className="text-body-sm font-medium text-navy-700 tabular-nums">
                {item.percent.toFixed(1)}%
              </span>
            </div>
          ))}
          {dataWithPercent.length > 6 && (
            <p className="text-caption text-neutral-500 pl-6">
              +{dataWithPercent.length - 6} more
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AllocationChart;

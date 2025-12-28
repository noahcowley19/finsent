'use client';

import React, { ReactNode, HTMLAttributes } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type GridCols = 1 | 2 | 3 | 4 | 5 | 6 | 12;
export type GridGap = 'none' | 'sm' | 'md' | 'lg' | 'xl';

export interface GridProps extends HTMLAttributes<HTMLDivElement> {
  /** Content */
  children: ReactNode;
  /** Default columns */
  cols?: GridCols;
  /** Columns at sm breakpoint */
  colsSm?: GridCols;
  /** Columns at md breakpoint */
  colsMd?: GridCols;
  /** Columns at lg breakpoint */
  colsLg?: GridCols;
  /** Columns at xl breakpoint */
  colsXl?: GridCols;
  /** Gap between items */
  gap?: GridGap;
}

// =============================================================================
// STYLES
// =============================================================================

const colStyles: Record<GridCols, string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
  5: 'grid-cols-5',
  6: 'grid-cols-6',
  12: 'grid-cols-12',
};

const gapStyles: Record<GridGap, string> = {
  none: 'gap-0',
  sm: 'gap-3',
  md: 'gap-4 lg:gap-5',
  lg: 'gap-5 lg:gap-6',
  xl: 'gap-6 lg:gap-8',
};

// =============================================================================
// GRID COMPONENT
// =============================================================================

export const Grid: React.FC<GridProps> = ({
  children,
  cols = 1,
  colsSm,
  colsMd,
  colsLg,
  colsXl,
  gap = 'md',
  className = '',
  ...props
}) => {
  const responsiveClasses = [
    colStyles[cols],
    colsSm && `sm:${colStyles[colsSm]}`,
    colsMd && `md:${colStyles[colsMd]}`,
    colsLg && `lg:${colStyles[colsLg]}`,
    colsXl && `xl:${colStyles[colsXl]}`,
  ].filter(Boolean).join(' ');

  return (
    <div
      className={`
        grid
        ${responsiveClasses}
        ${gapStyles[gap]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// BENTO GRID
// =============================================================================

export interface BentoGridProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const BentoGrid: React.FC<BentoGridProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4
        gap-4 lg:gap-5
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

export interface BentoItemProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Span multiple columns */
  colSpan?: 1 | 2 | 3 | 4;
  /** Span multiple rows */
  rowSpan?: 1 | 2;
}

export const BentoItem: React.FC<BentoItemProps> = ({
  children,
  colSpan = 1,
  rowSpan = 1,
  className = '',
  ...props
}) => {
  const colSpanStyles: Record<number, string> = {
    1: '',
    2: 'md:col-span-2',
    3: 'md:col-span-2 lg:col-span-3',
    4: 'md:col-span-2 lg:col-span-3 xl:col-span-4',
  };

  const rowSpanStyles: Record<number, string> = {
    1: '',
    2: 'row-span-2',
  };

  return (
    <div
      className={`
        ${colSpanStyles[colSpan]}
        ${rowSpanStyles[rowSpan]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

export default Grid;

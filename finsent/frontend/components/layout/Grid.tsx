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

// =============================================================================
// GRID ITEM
// =============================================================================

export interface GridItemProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  colSpan?: number;
  rowSpan?: number;
}

export const GridItem: React.FC<GridItemProps> = ({
  children,
  colSpan,
  rowSpan,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`
        ${colSpan ? `col-span-${colSpan}` : ''}
        ${rowSpan ? `row-span-${rowSpan}` : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// STACK
// =============================================================================

export interface StackProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  direction?: 'row' | 'col';
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between';
  spacing?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl';
}

const stackSpacing: Record<string, string> = {
  none: 'gap-0',
  xs: 'gap-2',
  sm: 'gap-4',
  md: 'gap-6',
  lg: 'gap-8',
  xl: 'gap-12',
};

export const Stack: React.FC<StackProps> = ({
  children,
  direction = 'col',
  align = 'stretch',
  justify = 'start',
  spacing = 'md',
  className = '',
  ...props
}) => {
  const directionClass = direction === 'row' ? 'flex-row' : 'flex-col';
  const alignClass = `items-${align}`;
  const justifyClass = `justify-${justify}`;

  return (
    <div
      className={`
        flex
        ${directionClass}
        ${alignClass}
        ${justifyClass}
        ${stackSpacing[spacing]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// TWO COLUMN
// =============================================================================

export interface TwoColumnProps extends HTMLAttributes<HTMLDivElement> {
  children: [ReactNode, ReactNode];
  ratio?: '1:1' | '1:2' | '2:1';
  align?: 'start' | 'center' | 'end';
}

export const TwoColumn: React.FC<TwoColumnProps> = ({
  children,
  ratio = '1:1',
  align = 'start',
  className = '',
  ...props
}) => {
  const ratioClasses: Record<string, string> = {
    '1:1': 'grid-cols-1 md:grid-cols-2',
    '1:2': 'grid-cols-1 md:grid-cols-3 [&>*:last-child]:md:col-span-2',
    '2:1': 'grid-cols-1 md:grid-cols-3 [&>*:first-child]:md:col-span-2',
  };

  const alignClass: Record<string, string> = {
    start: 'items-start',
    center: 'items-center',
    end: 'items-end',
  };

  return (
    <div
      className={`
        grid gap-8 lg:gap-12
        ${ratioClasses[ratio]}
        ${alignClass[align]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </div>
  );
};

export default Grid;
